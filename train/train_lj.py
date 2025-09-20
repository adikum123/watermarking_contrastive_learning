"""
Train audio watermarking model on part of the dataset
"""

import argparse
import json
import os
import warnings
import logging
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# ------------------ Logging Setup ------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()
# Also print to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# ------------------ Warnings ------------------
warnings.filterwarnings(
    "ignore", message=".*TorchCodec's decoder.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

# ------------------ Config ------------------
data_source = "ljspeech"

with open("config/train.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process_lj.yaml", "r") as f:
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

# ------------------ Dataset ------------------
batch_size = train_config["optimize"]["batch_size"]
train_ds, val_ds, test_ds = get_datasets(
    contrastive=False,
    process_config=process_config,
    take_part=False,
    dataset_type="ljspeech",
)

# Safe prefetching DataLoader setup
num_workers = min(4, os.cpu_count())  # 4 workers or # of CPUs if less
prefetch_factor = 2  # Each worker preloads 2 batches

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,   # Keeps workers alive between epochs
    pin_memory=True,           # Faster GPU transfer
    prefetch_factor=prefetch_factor,
    timeout=60,                # Avoid hangs on slow data
)

val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    timeout=60,
)

test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    timeout=60,
)

# ------------------ Device ------------------
device = torch.device("cpu") if args.use_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ------------------ Models ------------------
embedder, decoder, discriminator = init_models(
    model_config=model_config,
    process_config=process_config,
    train_config=train_config,
    device=device,
)

# ------------------ Loss ------------------
loss = WatermarkLoss(
    lambda_e=train_config["optimize"]["lambda_e"],
    lambda_m=train_config["optimize"]["lambda_m"],
    lambda_a=train_config["optimize"]["lambda_a"],
    lambda_cl=train_config["optimize"].get("lambda_cl", 0.0),
    adversarial=train_config["adv"],
    contrastive=False,
    contrastive_loss_type=None,
)

# ------------------ Optimizers and schedulers ------------------
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

logger.info(f"Training with params:\n{json.dumps(train_config, indent=4)}\nLength of train dataset: {len(train_ds)}")

# ------------------ Checkpoints ------------------
checkpoint_dir = os.path.join("model_ckpts", "lj")
os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 0
start_batch = 0
best_acc = None
best_pesq = None

if args.ckpt_path:
    logger.info(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    embedder.load_state_dict(checkpoint["embedder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if train_config["adv"] and checkpoint["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    em_de_opt.load_state_dict(checkpoint["em_de_opt_state_dict"])
    if train_config["adv"] and checkpoint["dis_opt_state_dict"] is not None:
        dis_opt.load_state_dict(checkpoint["dis_opt_state_dict"])
    start_epoch = checkpoint["epoch"]
    start_batch = checkpoint.get("batch_idx", 0)
    best_acc = checkpoint.get("average_acc", None)
    best_pesq = checkpoint.get("average_pesq", None)

# ------------------ Metric history ------------------
metric_history = {
    "train_loss": [],
    "train_pesq": [],
    "train_acc_identity": [],
    "train_acc_distorted": [],
    "val_loss": [],
    "val_pesq": [],
    "val_acc_identity": [],
}

# ------------------ Training Loop ------------------
for epoch in range(start_epoch, train_config["iter"]["epoch"] + 1):
    train_metrics = MetricsTracker(name="train")

    for i, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1} [Train]", file=sys.stdout)):
        if i < start_batch:
            logger.info(f"Skipping batch {i} < {start_batch} to resume from checkpoint")
            continue

        embedder.train()
        decoder.train()
        discriminator.train()

        wav, msg = prepare_batch(batch, train_config["watermark"]["length"], device)
        curr_bs = wav.shape[0]

        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        pesq_score = pesq_score_batch(
            wav.squeeze(1),
            embedded.squeeze(1),
            sr=process_config["audio"]["sample_rate"],
        )

        dis_output_embedded = None
        if train_config["adv"]:
            dis_output_embedded = discriminator(embedded)

        sum_loss = loss(
            embedded=embedded,
            decoded=decoded,
            wav=wav,
            msg=msg,
            discriminator_output=dis_output_embedded,
        )
        sum_loss.backward()

        if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
            em_de_opt.step()
            em_de_opt.zero_grad()
            if train_config["adv"]:
                dis_opt.step()
                dis_opt.zero_grad()

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

        train_metrics.update(
            loss=sum_loss.item(),
            pesq=pesq_score,
            decoded=decoded,
            msg=msg,
            batch_size=curr_bs,
        )

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(json.dumps(train_metrics.summary(), indent=4))

    # ------------------ Validation ------------------
    with torch.no_grad():
        embedder.eval()
        decoder.eval()
        discriminator.eval()

        val_metrics = MetricsTracker(name="val")
        for i, batch in enumerate(tqdm(val_dl, desc=f"Epoch {epoch+1} [Val]", file=sys.stdout)):
            wav, msg = prepare_batch(batch, train_config["watermark"]["length"], device)
            curr_bs = wav.shape[0]

            embedded, carrier_wateramrked = embedder(wav, msg)
            decoded = decoder(embedded)

            pesq_score = pesq_score_batch(
                wav.squeeze(1),
                embedded.squeeze(1),
                sr=process_config["audio"]["sample_rate"],
            )

            dis_output_embedded = None
            if train_config["adv"]:
                dis_output_embedded = discriminator(embedded.detach())

            sum_loss = loss(
                embedded=embedded,
                decoded=decoded,
                wav=wav,
                msg=msg,
                discriminator_output=dis_output_embedded,
            )

            val_metrics.update(
                loss=sum_loss.item(),
                pesq=pesq_score,
                decoded=decoded,
                msg=msg,
                batch_size=curr_bs,
            )

            if (i + 1) % 100 == 0 or i == 0:
                logger.info(json.dumps(val_metrics.summary(), indent=4))

    # ------------------ Save checkpoint ------------------
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
            checkpoint_dir, f"wm_model_lj_pesq_{curr_pesq}_acc_{curr_acc}.pt"
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
        logger.info(f"Checkpoint saved: {checkpoint_path} | Acc: {curr_acc}, PESQ: {curr_pesq}")

    em_de_sch.step()
    if train_config["adv"]:
        dis_sch.step()

    loss.schedule_lambdas(epoch=epoch + 1)
    logger.info(f"Decreased lambda m to {loss.lambda_m}")

    # ------------------ Store metrics ------------------
    train_summary = train_metrics.summary()
    val_summary = val_metrics.summary()

    metric_history["train_loss"].append(train_summary["loss"])
    metric_history["train_pesq"].append(train_summary["pesq"])
    metric_history["train_acc_identity"].append(train_summary["avg_acc_identity"])
    metric_history["train_acc_distorted"].append(train_summary.get("avg_acc_distorted", 0))
    metric_history["val_loss"].append(val_summary["loss"])
    metric_history["val_pesq"].append(val_summary["pesq"])
    metric_history["val_acc_identity"].append(val_summary["avg_acc"])

    # ------------------ Plot ------------------
    os.makedirs("train_plots", exist_ok=True)
    save_path = os.path.join("train_plots", "wm_train_plot_reg_models.png")
    epochs_range = range(1, len(metric_history["train_loss"]) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, metric_history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, metric_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, metric_history["train_pesq"], label="Train PESQ")
    plt.plot(epochs_range, metric_history["val_pesq"], label="Val PESQ")
    plt.xlabel("Epoch")
    plt.ylabel("PESQ")
    plt.title("PESQ")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, metric_history["train_acc_identity"], label="Train Acc Identity")
    plt.plot(epochs_range, metric_history["train_acc_distorted"], label="Train Acc Distorted")
    plt.plot(epochs_range, metric_history["val_acc_identity"], label="Val Acc Identity")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
