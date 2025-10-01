"""
Contrastive pre train the decoder
"""

import json
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import torch
import yaml
from torch.optim.lr_scheduler import StepLR

from loss.contrastive_loss import ContrastiveLoss
from model.cl_pt_decoder import ContrastiveDecoder
from model.utils import create_loader, get_datasets

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

# ------------------ Config ------------------
with open("config/contrastive_pretrain.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process_lj.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/cl_pt_decoder.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# ------------------ Dataset ------------------
batch_size = train_config["optimize"]["batch_size"]
train_ds, val_ds, _ = get_datasets(
    contrastive=True,
    process_config=process_config,
    take_part=False,
    dataset_type="ljspeech",
)
train_dl = create_loader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_dl = create_loader(dataset=val_ds, batch_size=batch_size, shuffle=False)


# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)
checkpoint_dir = os.path.join("model_ckpts", "lj", "cl_pretrain_decoder")

# ------------------ Models ------------------
print(model_config)
cl_decoder = ContrastiveDecoder(
    process_config=process_config,
    model_config=model_config,
    msg_length=train_config["watermark"]["length"],
    win_dim=model_config["audio"]["win_dim"],
    mode="contrastive_pretrain"
)

# ------------------ Loss ------------------
loss = ContrastiveLoss(loss_type=train_config["contrastive"]["loss_type"])

# ------------------ Optimizers and schedulers ------------------
optimizer = torch.optim.Adam(
    cl_decoder.get_train_params(),
    lr=train_config["optimize"]["lr"],
    weight_decay=train_config["optimize"]["weight_decay"],
    betas=train_config["optimize"]["betas"],
    eps=train_config["optimize"]["eps"],
)
scheduler = StepLR(
    optimizer,
    step_size=train_config["optimize"]["step_size"],
    gamma=train_config["optimize"]["gamma"],
)

logger.info(
    "Training with params:\n%s\nLength of train dataset: %s",
    json.dumps(train_config, indent=4),
    len(train_ds)
)

# ------------------ Metric history ------------------
metric_history = {
    "train_cl_loss": [],
    "train_val_loss": []
}

# ------------------ Training Loop ------------------
for epoch in range(0, train_config["iter"]["epoch"]):
    cl_decoder.train()
    logger.info("Epoch: %s", epoch + 1)
    total_train_loss, total_train_num = 0, 0
    for i, batch in enumerate(train_dl):
        # get augmented views and contrastive loss
        curr_bs = batch["wav"].shape[0]
        aug_view1, aug_view2 = batch["augmented_views"]
        feat_view1 = cl_decoder.get_features(x=aug_view1)
        feat_view2 = cl_decoder.get_features(x=aug_view2)
        cl_loss = loss(
            feat_view1.squeeze(1), feat_view2.squeeze(1)
        )
        cl_loss.backward()

        # update trackers
        total_train_loss += cl_loss.item() * curr_bs
        total_train_num += curr_bs

        if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % 100 == 0 or i == 0:
            logger.info("Processed %s batches", i + 1)
            logger.info("Curr average loss: %s", total_train_loss / total_train_num)

    avg_train_loss = total_train_loss / total_train_num
    # ------------------ Validation ------------------
    with torch.no_grad():
        cl_decoder.eval()
        total_val_loss, total_val_num = 0, 0
        for i, batch in enumerate(val_dl):
            # get augmented views and contrastive loss
            curr_bs = batch["wav"].shape[0]
            feat_view1, feat_view2 = batch["augmented_views"]
            cl_loss = loss(
                feat_view1.squeeze(0), feat_view2.squeeze(0)
            )

            # update trackers
            total_val_loss += cl_loss.item() * curr_bs
            total_val_num += curr_bs

            if (i + 1) % 100 == 0 or i == 0:
                logger.info("Processed %s batches", i + 1)
                logger.info("Curr average loss: %s", total_val_loss / total_val_num)

    avg_val_loss = total_val_loss / total_val_num

    # ------------------ Save checkpoint ------------------
    checkpoint_path = os.path.join(
        checkpoint_dir,
        "cl_pt_decoder_train_loss_{}_val_loss_{}_epoch_{}.pt".format(
            avg_train_loss,
            avg_val_loss,
            epoch + 1
        )
    )
    torch.save(
        {
            "epoch": epoch,
            "batch_idx": i,
            "decoder_state_dict": cl_decoder.state_dict(),
            "decoder_opt_state_dict": optimizer.state_dict()
        },
        checkpoint_path,
    )

    scheduler.step()

    # ------------------ Save metric history ------------------
    metric_history["train_cl_loss"].append(avg_train_loss)
    metric_history["train_val_loss"].append(avg_val_loss)

    # ------------------ Plot losses ------------------
    plt.figure(figsize=(8, 5))
    plt.plot(metric_history["train_cl_loss"], label="Train Loss")
    plt.plot(metric_history["train_val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # overwrite the previous plot file
    loss_plot_path = "logs/loss_curve.png"
    plt.savefig(loss_plot_path)
    plt.close()  # close the figure to free memory
