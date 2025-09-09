from itertools import chain

import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from torch.optim.lr_scheduler import StepLR

from data.contrastive_dataset import ContrastiveAudioDataset
from data.dataset import AudioDataset
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

    return float(np.sum(scores))


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
        try:
            score = pesq(sr, ref_np, deg_np, mode)
        except Exception:
            # PESQ can throw if inputs are too short or silent
            print("PESQ computation failed, assigning score of 0.0")
            continue
        scores.append(score)

    return float(np.mean(scores))


def get_datasets(contrastive=False, **kwargs):
    """
    Utility function to get datasets.
    """
    if contrastive:
        train_ds = ContrastiveAudioDataset(
            kwargs["process_config"],
            split="train",
            take_num=5000 if kwargs.get("take_part", False) else None,
        )
        val_ds = ContrastiveAudioDataset(
            kwargs["process_config"],
            split="val",
            take_num=1000 if kwargs.get("take_part", False) else None,
        )
        test_ds = ContrastiveAudioDataset(
            kwargs["process_config"],
            split="test",
            take_num=1000 if kwargs.get("take_part", False) else None,
        )
        return train_ds, val_ds, test_ds
    train_ds = AudioDataset(
        kwargs["process_config"],
        split="train",
        take_num=5000 if kwargs.get("take_part", False) else None,
    )
    val_ds = AudioDataset(
        kwargs["process_config"],
        split="val",
        take_num=1000 if kwargs.get("take_part", False) else None,
    )
    test_ds = AudioDataset(
        kwargs["process_config"],
        split="test",
        take_num=1000 if kwargs.get("take_part", False) else None,
    )
    return train_ds, val_ds, test_ds


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
    embedding_dim = model_config["dim"]["embedding"]
    msg_length = train_config["watermark"]["length"]
    win_dim = model_config["audio"]["win_dim"]

    # Models
    embedder = Embedder(
        process_config,
        model_config,
        msg_length,
        win_dim,
        embedding_dim,
    ).to(device)
    decoder = Decoder(
        process_config,
        model_config,
        msg_length,
        win_dim,
        embedding_dim,
    ).to(device)
    discriminator = Discriminator(process_config).to(device)
    return embedder, decoder, discriminator


def init_optimizers(embedder, decoder, discriminator, train_config):
    """
    Utility function to init optimizers
    """
    em_de_opt = torch.optim.Adam(
        chain(embedder.parameters(), decoder.parameters()),
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
