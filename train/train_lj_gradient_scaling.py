"""
Train audio watermarking model on part of the dataset
"""

import argparse
import json
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import yaml

from loss.loss_gradient_scaling import LossGradientScaling
from model.metrics_tracker import MetricsTracker
from model.utils import (create_loader, get_datasets, init_models,
                         init_optimizers, pesq_score_batch, prepare_batch,
                         save_model)

# ------------------ Logging Setup ------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()

# Also print to console with flushing
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
console_handler.flush = sys.stdout.flush  # ensure flushing
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

# ------------------ Seed --------------------
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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
train_ds, val_ds, _ = get_datasets(
    contrastive=False,
    process_config=process_config,
    take_part=False,
    dataset_type="ljspeech",
)
train_dl = create_loader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_dl = create_loader(dataset=val_ds, batch_size=batch_size, shuffle=False)


# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)

# ------------------ Models ------------------
embedder, decoder, _ = init_models(
    model_config=model_config,
    process_config=process_config,
    train_config=train_config,
    device=device,
)

# ------------------ Loss ------------------
loss = LossGradientScaling(
    contrastive=False,
    adversarial=False,
    clip_grad_norm=None,  # or a float if you want gradient clipping
    beta=1,  # scaling exponent
    eps=1e-6,
)

# ------------------ Optimizers and schedulers ------------------
em_de_opt, dis_opt = init_optimizers(
    embedder=embedder,
    decoder=decoder,
    discriminator=None,
    train_config=train_config,
    finetune=False,
)

logger.info(
    "Training with params:\n%s\nLength of train dataset: %s",
    json.dumps(train_config, indent=4),
    len(train_ds),
)

# ------------------ Checkpoints ------------------
checkpoint_dir = os.path.join("model_ckpts", "lj")
os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 0
start_batch = 0
best_acc = None
best_pesq = None

if args.ckpt_path:
    logger.info("Loading checkpoint from: %s", {args.ckpt_path})
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    embedder.load_state_dict(checkpoint["embedder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    em_de_opt.load_state_dict(checkpoint["em_de_opt_state_dict"])
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
    logger.info("Epoch: %s", epoch + 1)
    for i, batch in enumerate(train_dl):
        if i < start_batch:
            continue

        embedder.train()
        decoder.train()

        wav, msg = prepare_batch(batch, train_config["watermark"]["length"], device)
        curr_bs = wav.shape[0]

        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        pesq_score = pesq_score_batch(
            wav.squeeze(1),
            embedded.squeeze(1),
            sr=process_config["audio"]["sample_rate"],
        )

        losses_dict = loss.compute_losses(
            embedded=embedded,
            decoded=decoded,
            wav=wav,
            msg=msg,
            discriminator_output=None,
        )
        params = list(embedder.parameters()) + list(decoder.parameters())
        loss.backward(losses_dict, params)

        if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
            em_de_opt.step()
            em_de_opt.zero_grad()

        train_metrics.update(
            loss=sum(loss.item() for loss in losses_dict.values()),
            pesq=pesq_score,
            decoded=decoded,
            msg=msg,
            batch_size=curr_bs,
        )

        if (i + 1) % 100 == 0 or i == 0:
            logger.info("Processed %s batches", i + 1)
            logger.info(json.dumps(train_metrics.summary(), indent=4))

    # ------------------ Validation ------------------
    with torch.no_grad():
        embedder.eval()
        decoder.eval()

        val_metrics = MetricsTracker(name="val")
        for i, batch in enumerate(val_dl):
            wav, msg = prepare_batch(batch, train_config["watermark"]["length"], device)
            curr_bs = wav.shape[0]

            embedded, carrier_wateramrked = embedder(wav, msg)
            decoded = decoder(embedded)

            pesq_score = pesq_score_batch(
                wav.squeeze(1),
                embedded.squeeze(1),
                sr=process_config["audio"]["sample_rate"],
            )

            losses_dict = loss.compute_losses(
                embedded=embedded,
                decoded=decoded,
                wav=wav,
                msg=msg,
                discriminator_output=None,
            )

            val_metrics.update(
                loss=sum(loss.item() for loss in losses_dict.values()),
                pesq=pesq_score,
                decoded=decoded,
                msg=msg,
                batch_size=curr_bs,
            )

            if (i + 1) % 100 == 0 or i == 0:
                logger.info("Processed %s batches", i + 1)
                logger.info(json.dumps(val_metrics.summary(), indent=4))

    # ------------------ Save checkpoint ------------------
    curr_acc = val_metrics.avg_acc_identity()
    curr_pesq = val_metrics.average_pesq()
    if (
        save_model(
            best_pesq=best_pesq,
            best_acc=best_acc,
            new_pesq=curr_pesq,
            new_acc=curr_acc,
            min_pesq=3.5,
            min_acc=0.95,
        )
        or True
    ):  # save all models
        best_acc, best_pesq = curr_acc, curr_pesq
        checkpoint_path = os.path.join(
            checkpoint_dir,
            "wm_model_lj_pesq_{:.2f}_acc_{:.2f}_dist_acc_{:.2f}_epoch_{}_gs.pt".format(
                curr_pesq, curr_acc, train_metrics.avg_acc_distorted(), epoch + 1
            ),
        )
        torch.save(
            {
                "epoch": epoch,
                "batch_idx": i,
                "embedder_state_dict": embedder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "em_de_opt_state_dict": em_de_opt.state_dict(),
                "average_acc": curr_acc,
                "average_pesq": curr_pesq,
            },
            checkpoint_path,
        )
        logger.info(
            "Checkpoint saved: %s | Acc: %s, PESQ: %s",
            checkpoint_path,
            curr_acc,
            curr_pesq,
        )

    # ------------------ Store metrics ------------------
    train_summary = train_metrics.summary()
    val_summary = val_metrics.summary()

    metric_history["train_loss"].append(train_summary["loss"])
    metric_history["train_pesq"].append(train_summary["pesq"])
    metric_history["train_acc_identity"].append(train_summary["avg_acc_identity"])
    metric_history["train_acc_distorted"].append(
        train_summary.get("avg_acc_distorted", 0)
    )
    metric_history["val_loss"].append(val_summary["loss"])
    metric_history["val_pesq"].append(val_summary["pesq"])
    metric_history["val_acc_identity"].append(val_summary["avg_acc"])

    # ------------------ Plot ------------------
    os.makedirs("train_plots", exist_ok=True)
    save_path = os.path.join("train_plots", "wm_train_plot.png")
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
    plt.plot(
        epochs_range, metric_history["train_acc_identity"], label="Train Acc Identity"
    )
    plt.plot(
        epochs_range, metric_history["train_acc_distorted"], label="Train Acc Distorted"
    )
    plt.plot(epochs_range, metric_history["val_acc_identity"], label="Val Acc Identity")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
