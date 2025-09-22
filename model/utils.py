import logging
from itertools import chain

import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

from data.lj_dataset import LjAudioDataset
from model.decoder import Decoder
from model.discriminator import Discriminator
from model.embedder import Embedder


def stoi_score_batch(ref_batch, deg_batch, fs):
    """
    Compute STOI for a batch of audio signals.

    Args:
        ref_batch: torch.Tensor of shape (batch, n_samples)
        deg_batch: torch.Tensor of shape (batch, n_samples)
        fs: sample rate (e.g. 16000)

    Returns:
        scores: list of floats, STOI per sample in batch
    """
    assert (
        ref_batch.shape == deg_batch.shape
    ), f"Reference and degraded must have same shape: {ref_batch.shape}, {deg_batch.shape}"

    scores = []
    for ref, deg in zip(ref_batch, deg_batch):
        ref = ref.detach().cpu().numpy().astype(float)
        deg = deg.detach().cpu().numpy().astype(float)

        # Truncate to min length in case of small mismatches
        min_len = min(len(ref), len(deg))
        ref, deg = ref[:min_len], deg[:min_len]

        score = stoi(ref, deg, fs, extended=False)
        scores.append(score)

    return float(np.mean(scores))


def pesq_score_batch(source_batch, target_batch, sr, mode="wb"):
    """
    Compute PESQ for a batch of audios.

    Args:
        source_batch (torch.Tensor): Shape (B, T) – reference audios
        target_batch (torch.Tensor): Shape (B, T) – degraded audios
        sr (int): Sampling rate of input signals
        mode (str): 'wb' for wideband, 'nb' for narrowband

    Returns:
        float: Sum of PESQ scores across the batch
    """
    assert (
        source_batch.shape == target_batch.shape
    ), "Source and target must have same shape"

    # Move tensors to CPU for resampling and PESQ computation
    source_batch = source_batch.detach().cpu()
    target_batch = target_batch.detach().cpu()

    # PESQ only supports 8kHz or 16kHz
    if sr not in (8000, 16000):
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        source_batch = resampler(source_batch)
        target_batch = resampler(target_batch)
        sr = 16000
        mode = "wb"  # force wideband for 16kHz

    scores = []
    for ref, deg in zip(source_batch, target_batch):
        ref_np = ref.numpy()
        deg_np = deg.numpy()
        # Assert they are 1D arrays
        assert ref_np.ndim == 1, f"Reference audio must be 1D, got shape {ref_np.shape}"
        assert deg_np.ndim == 1, f"Degraded audio must be 1D, got shape {deg_np.shape}"
        try:
            score = pesq(sr, ref_np, deg_np, mode)
        except Exception:
            # PESQ can throw if inputs are too short or silent
            print("PESQ computation failed, assigning score of 0.0")
            continue
        scores.append(score)

    return float(np.mean(scores))


def get_datasets(dataset_type, contrastive, **kwargs):
    """
    Utility function to get datasets.
    """
    assert dataset_type == "ljspeech", "Contrastive not yet implemented and must be ljspeech ds"
    if dataset_type == "ljspeech":
        print("Loading precomputed ljspeech datasets")
        train_ds = LjAudioDataset(
            split="train",
            process_config=kwargs["process_config"],
            contrastive=contrastive,
        )
        val_ds = LjAudioDataset(
            split="val",
            process_config=kwargs["process_config"],
            contrastive=contrastive,
        )
        test_ds = LjAudioDataset(
            split="test",
            process_config=kwargs["process_config"],
            contrastive=contrastive
        )
        return train_ds, val_ds, test_ds
    raise ValueError("Must use lj speech dataset")


def create_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers=0,
    prefetch_factor=2,
    timeout=0,
    persistent_workers=False,
    drop_last=False,
    name="train"
):
    """
    Create a DataLoader with consistent logging and worker setup.
    """

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = batch_size * num_gpus
    workers_per_gpu = num_workers
    total_workers = workers_per_gpu * num_gpus

    logger.info(
        f":package: {name} loader ready → "
        f"samples={len(dataset)}, batch_size={effective_batch_size}, "
        f"total_workers={total_workers}, workers_per_gpu={workers_per_gpu}, "
        f"num_gpus={num_gpus}, shuffle={shuffle}"
    )

    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=total_workers,
        persistent_workers=persistent_workers if total_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if total_workers > 0 else None,
        timeout=timeout,
        drop_last=drop_last,
    )


def save_model(best_pesq, best_acc, new_pesq, new_acc, min_pesq=4.0, min_acc=0.9):
    """
    Decide whether to save a model based on PESQ and accuracy tradeoff.

    Args:
        best_pesq (float or None): Best PESQ seen so far.
        best_acc (float or None): Best accuracy seen so far.
        new_pesq (float): PESQ of current model.
        new_acc (float): Accuracy of current model.
        min_pesq (float): Minimum acceptable PESQ to save model.
        min_acc (float): Minimum acceptable accuracy to save model.

    Returns:
        bool: True if model should be saved.
    """
    # If no previous best, always save
    if best_pesq is None or best_acc is None:
        return True

    # Check if new metrics are “good enough”
    pesq_ok = new_pesq >= min_pesq
    acc_ok = new_acc >= min_acc

    # Save only if one metric improves and the other is still good enough
    if new_pesq > best_pesq and acc_ok:
        return True
    if new_acc > best_acc and pesq_ok:
        return True

    # Otherwise, do not save
    return False


def init_models(model_config, train_config, process_config, device):
    """
    Utility function to init models
    """
    # Model config
    msg_length = train_config["watermark"]["length"]
    win_dim = model_config["audio"]["win_dim"]

    # Models
    embedder = Embedder(
        process_config=process_config,
        model_config=model_config,
        msg_length=msg_length,
        win_dim=win_dim,
    ).to(device)
    decoder = Decoder(
        process_config=process_config,
        model_config=model_config,
        msg_length=msg_length,
        win_dim=win_dim,
    ).to(device)
    discriminator = Discriminator(process_config).to(device)
    return embedder, decoder, discriminator


def init_optimizers(embedder, decoder, discriminator, train_config, finetune):
    """
    Utility function to init optimizers
    """
    em_de_opt = torch.optim.Adam(
        chain(embedder.parameters(), decoder.get_train_params(finetune=finetune)),
        lr=train_config["optimize"]["lr"],
        weight_decay=train_config["optimize"]["weight_decay"],
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
    )
    dis_opt = None
    if train_config["adv"]:
        dis_opt = torch.optim.Adam(
            discriminator.parameters(),
            lr=train_config["optimize"]["lr"],
            weight_decay=train_config["optimize"]["weight_decay"],
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
        )
    return em_de_opt, dis_opt


def init_schedulers(em_de_opt, dis_opt, train_config):
    """
    Utility function to init schedulers
    """
    em_de_scheduler = StepLR(
        em_de_opt,
        step_size=train_config["optimize"]["step_size"],
        gamma=train_config["optimize"]["gamma"],
    )
    dis_scheduler = None
    if train_config["adv"]:
        dis_scheduler = StepLR(
            dis_opt,
            step_size=train_config["optimize"]["step_size"],
            gamma=train_config["optimize"]["gamma"],
        )
    return em_de_scheduler, dis_scheduler


def prepare_batch(batch, msg_length, device):
    """
    Utility function to prepare batch
    """
    wav = batch["wav"].to(device)
    curr_bs = wav.shape[0]
    msg = torch.randint(0, 2, (curr_bs, 1, msg_length), device=device).float() * 2 - 1
    return wav, msg


def accuracy(decoded, msg):
    """
    Compute accuracy of decoded watermark against original message.
    """
    return (decoded >= 0).eq(msg >= 0).sum().float().item()


def lambda_schedule(epoch_idx, total_epochs):
    """
    Compute (lambda_e, lambda_m) for the given epoch index, scaled to total_epochs.

    Strategy:
      - Phase 1 (first 50% of epochs): detector-heavy, embedder weight ramps up.
      - Phase 2 (second 50% of epochs): flip emphasis toward fidelity.

    Args:
        epoch_idx (int): 1-based epoch number (1..total_epochs).
        total_epochs (int): total planned epochs.

    Returns:
        (lambda_e, lambda_m)
    """

    # clamp to [1, total_epochs]
    e = max(1, min(total_epochs, int(epoch_idx)))

    # Split into two phases (50/50 by default)
    half = total_epochs // 2
    if e <= half:
        # Phase 1: lambda_e grows, lambda_m shrinks
        progress = (e - 1) / max(1, (half - 1))
        lambda_e = 1.0 + progress * (2.8 - 1.0)  # 1.0 -> 2.8
        lambda_m = 2.0 - progress * (2.0 - 1.1)  # 2.0 -> 1.1
        return lambda_e, lambda_m
    # Phase 2: continue shifting toward fidelity
    progress = (e - half) / max(1, (total_epochs - half))
    lambda_e = 2.8 + progress * (3.5 - 2.8)  # 2.8 -> 3.5
    lambda_m = 1.1 - progress * (1.1 - 0.5)  # 1.1 -> 0.5
    return lambda_e, lambda_m


def get_model_average_pesq_dataset(embedder, dataset, msg_length, device, sr):
    """
    Computes average pesq on dataset
    """
    total_pesq = 0
    print("Computing average pesq")
    for item in tqdm(dataset):
        # embed message
        wav = item["wav"].unsqueeze(0).to(device)
        curr_bs = wav.shape[0]
        msg = (
            torch.randint(0, 2, (curr_bs, 1, msg_length), device=device).float() * 2 - 1
        ).to(device)
        embedded, _ = embedder(wav, msg)

        # compute pesq
        total_pesq += pesq_score_batch(
            source_batch=wav.squeeze(0), target_batch=embedded.squeeze(0), sr=sr
        )

    return total_pesq / len(dataset)


def get_model_average_stoi_dataset(embedder, dataset, msg_length, device, sr):
    """
    Computes average stoi on dataset
    """
    total_stoi = 0
    print("Computing average stoi")
    for item in tqdm(dataset):
        # embed message
        wav = item["wav"].unsqueeze(0).to(device)
        curr_bs = wav.shape[0]
        msg = (
            torch.randint(0, 2, (curr_bs, 1, msg_length), device=device).float() * 2 - 1
        ).to(device)
        embedded, _ = embedder(wav, msg)

        # compute stoi
        total_stoi += stoi_score_batch(
            ref_batch=wav.squeeze(0), deg_batch=embedded.squeeze(0), fs=sr
        )

    return total_stoi / len(dataset)


def truncate_or_pad_np(audio, target_len):
    """
    Truncates or pads audio as np
    """
    curr_len = len(audio)
    if curr_len > target_len:
        return audio[:target_len]
    if curr_len < target_len:
        return np.pad(audio, (0, target_len - curr_len))
    return audio


def load_from_ckpt(ckpt, model_config, train_config, process_config, device):
    """
    Loads embedder and decoder from ckpt
    """
    # Model config
    msg_length = train_config["watermark"]["length"]
    win_dim = model_config["audio"]["win_dim"]

    # Models
    embedder = Embedder(
        process_config=process_config,
        model_config=model_config,
        msg_length=msg_length,
        win_dim=win_dim,
    ).to(device)
    decoder = Decoder(
        process_config=process_config,
        model_config=model_config,
        msg_length=msg_length,
        win_dim=win_dim,
    ).to(device)

    # load ckpt
    checkpoint = torch.load(ckpt, map_location=device)
    embedder.load_state_dict(checkpoint["embedder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    return embedder, decoder
