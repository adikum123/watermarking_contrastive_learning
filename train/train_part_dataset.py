"""
Train audio watermarking model on part of the dataset
"""

import argparse
import json
import os
import warnings

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss.loss import WatermarkLoss
from model.metrics_tracker import MetricsTracker
from model.utils import (
    get_datasets,
    init_models,
    init_optimizers,
    init_schedulers,
    pesq_score_batch,
    prepare_batch,
    save_model,
)

warnings.filterwarnings(
    "ignore", message=".*TorchCodec's decoder.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

# Load config
with open("config/train_part.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

parser = argparse.ArgumentParser(
    description="Train audio watermarking model with contrastive learning"
)
parser.add_argument(
    "--ckpt_path", type=str, default="", help="Path to checkpoint to resume training"
)
parser.add_argument("--use_cpu", action="store_true", help="Use cpu for training")
args = parser.parse_args()

# Dataset setup
batch_size = train_config["optimize"]["batch_size"]
train_ds, val_ds, test_ds = get_datasets(
    contrastive=False, process_config=process_config, take_part=True
)

# Dataloader setup
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=None,
    timeout=0,
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=None,
    timeout=0,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=None,
    timeout=0,
)

# Device
if args.use_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Models
embedder, decoder, discriminator = init_models(
    model_config=model_config,
    process_config=process_config,
    train_config=train_config,
    device=device,
)

# init loss class
loss = WatermarkLoss(
    lambda_e=train_config["optimize"]["lambda_e"],
    lambda_m=train_config["optimize"]["lambda_m"],
    lambda_a=train_config["optimize"]["lambda_a"],
    lambda_cl=train_config["optimize"].get("lambda_cl", 0.0),
    adversarial=train_config["adv"],
    contrastive=False,
    contrastive_loss_type=None,
)

# get optimizers and schedulers
em_de_opt, dis_opt = init_optimizers(
    embedder=embedder,
    decoder=decoder,
    discriminator=discriminator,
    train_config=train_config,
    finetune=False,
)
em_de_sch, dis_sch = init_schedulers(
    em_de_opt=em_de_opt,
    dis_opt=dis_opt,
    train_config=train_config,
)

print(
    f"Training with params:\n{json.dumps(train_config, indent=4)}\nLength of train dataset: {len(train_ds)}"
)

# init checkpoint dir
checkpoint_dir = "model_ckpts"
os.makedirs(checkpoint_dir, exist_ok=True)

# load from checkpoint if it exists
start_epoch = 0
start_batch = 0
best_acc = None
best_pesq = None
if args.ckpt_path:
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)

    embedder.load_state_dict(checkpoint["embedder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if train_config["adv"] and checkpoint["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    em_de_opt.load_state_dict(checkpoint["em_de_opt_state_dict"])
    if train_config["adv"] and checkpoint["dis_opt_state_dict"] is not None:
        dis_opt.load_state_dict(checkpoint["dis_opt_state_dict"])

    # extract start training epoch and batch
    start_epoch = checkpoint["epoch"]
    start_batch = checkpoint.get("batch_idx", 0)

    # extract val data
    best_acc = checkpoint.get("average_acc", None)
    best_pesq = checkpoint.get("average_pesq", None)

if start_batch == (len(train_ds) // batch_size) - 1:
    start_batch = 0
    start_epoch += 1

# start training
for epoch in range(start_epoch, train_config["iter"]["epoch"] + 1):
    # set params for tracking
    train_metrics = MetricsTracker(name="train")

    # set pbar for progress
    tbar = tqdm(
        enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1} [Train]"
    )
    for i, batch in tbar:
        # skip until we reach batch where we stopped
        if i < start_batch:
            print(f"Skipping batch {i} < {start_batch} to resume from checkpoint")
            continue

        # set all models to train mode
        embedder.train()
        decoder.train()
        discriminator.train()

        # get current audio and watermark message
        wav, msg = prepare_batch(batch, train_config["watermark"]["length"], device)
        curr_bs = wav.shape[0]

        # get the embedded audio, carrier watermarked audio and decoded message
        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        # compute pesq score on embedded
        pesq_score = pesq_score_batch(
            wav.squeeze(1),
            embedded.squeeze(1),
            sr=process_config["audio"]["sample_rate"],
        )

        # discriminator loss - first classify the embedded as true
        dis_output_embedded = None
        if train_config["adv"]:
            dis_output_embedded = discriminator(embedded)

        # backward pass
        sum_loss = loss(
            embedded=embedded,
            decoded=decoded,
            wav=wav,
            msg=msg,
            discriminator_output=dis_output_embedded,
        )
        sum_loss.backward()

        # perform gradient accumulation
        if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
            em_de_opt.step()
            em_de_opt.zero_grad()
            if train_config["adv"]:
                dis_opt.step()
                dis_opt.zero_grad()

        # backward pass on discriminator
        if train_config["adv"]:
            loss.discriminator_loss(
                curr_bs=curr_bs,
                device=device,
                discriminator=discriminator,
                embedded=embedded,
                wav=wav,
            )
            if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
                dis_opt.step()
                dis_opt.zero_grad()

        # update metrics
        train_metrics.update(
            loss=sum_loss.item(),
            pesq=pesq_score,
            decoded=decoded,
            msg=msg,
            batch_size=curr_bs,
        )
        tbar.set_postfix(
            {**train_metrics.summary(), "lr": em_de_opt.param_groups[0]["lr"]}
        )

    # val step
    with torch.no_grad():

        # set all models to eval
        embedder.eval()
        decoder.eval()
        discriminator.eval()

        # set params for tracking
        val_metrics = MetricsTracker(name="val")

        # set pbar for progress tracking
        vbar = tqdm(
            val_dl,
            total=len(val_dl),
            desc=f"Epoch {epoch+1} [Val]",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in vbar:
            # get current audio and watermark message
            wav, msg = prepare_batch(batch, train_config["watermark"]["length"], device)
            curr_bs = wav.shape[0]

            # get the embedded audio, carrier watermarked audio and decoded message
            embedded, carrier_wateramrked = embedder(wav, msg)
            decoded = decoder(embedded)

            # compute pesq score on batch
            pesq_score = pesq_score_batch(
                wav.squeeze(1),
                embedded.squeeze(1),
                sr=process_config["audio"]["sample_rate"],
            )

            # discriminator loss - first classify the embedded as true
            dis_output_embedded = None
            if train_config["adv"]:
                dis_output_embedded = discriminator(embedded.detach())

            # sum loss
            sum_loss = loss(
                embedded=embedded,
                decoded=decoded,
                wav=wav,
                msg=msg,
                discriminator_output=dis_output_embedded,
            )

            # update metrics
            val_metrics.update(
                loss=sum_loss.item(),
                pesq=pesq_score,
                decoded=decoded,
                msg=msg,
                batch_size=curr_bs,
            )

            # set pbar desc
            vbar.set_postfix(val_metrics.summary())

    # Save model checkpoint
    curr_acc = val_metrics.avg_acc_identity()
    curr_pesq = val_metrics.average_pesq()
    if save_model(
        best_pesq=best_pesq,
        best_acc=best_acc,
        new_pesq=curr_pesq,
        new_acc=curr_acc,
        min_pesq=3.5,
        min_acc=0.85,
    ):
        best_acc, best_pesq = curr_acc, curr_pesq
        checkpoint_path = os.path.join(
            checkpoint_dir, f"wm_model_pesq_{curr_pesq}_acc_{curr_acc}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "batch_idx": i,
                "embedder_state_dict": embedder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "discriminator_state_dict": (
                    discriminator.state_dict() if train_config["adv"] else None
                ),
                "em_de_opt_state_dict": em_de_opt.state_dict(),
                "dis_opt_state_dict": (
                    dis_opt.state_dict() if train_config["adv"] else None
                ),
                "average_acc": curr_acc,
                "average_pesq": curr_pesq,
            },
            checkpoint_path,
        )
        print(
            f"Checkpoint saved: {checkpoint_path} with new best average accuracy: {curr_acc} and pesq: {curr_pesq}"
        )

    em_de_sch.step()
    if train_config["adv"]:
        dis_sch.step()

    loss.schedule_lambdas(epoch=epoch + 1)
    print(f"Decreased lambda m to {loss.lambda_m}")
