import argparse
import json
import os
import warnings
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.contrastive_dataset import ContrastiveAudioDataset
from loss.contrastive_loss import ContrastiveLoss
from model.decoder import Decoder
from model.discriminator import Discriminator
from model.embedder import Embedder
from model.utils import stoi_score_batch

warnings.filterwarnings(
    "ignore", message=".*TorchCodec's decoder.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

# Load config
with open("config/train_part_cl.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

parser = argparse.ArgumentParser(
    description="Train audio watermarking model with contrastive learning"
)
parser.add_argument(
    "--save_ckpt", action="store_true", help="Store model ckpts on google drive"
)
parser.add_argument(
    "--ckpt_path", type=str, default="", help="Path to checkpoint to resume training"
)
args = parser.parse_args()

# Dataset setup
batch_size = train_config["optimize"]["batch_size"]
train_ds = ContrastiveAudioDataset(process_config, split="train", take_num=5000)
val_ds = ContrastiveAudioDataset(process_config, split="val", take_num=1000)
test_ds = ContrastiveAudioDataset(process_config, split="test", take_num=1000)
# Dataloader setup
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# Model config
embedding_dim = model_config["dim"]["embedding"]
nlayers_encoder = model_config["layer"]["nlayers_encoder"]
nlayers_decoder = model_config["layer"]["nlayers_decoder"]
msg_length = train_config["watermark"]["length"]
win_dim = model_config["audio"]["win_dim"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Models
embedder = Embedder(
    process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder
).to(device)
decoder = Decoder(
    process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder
).to(device)
discriminator = Discriminator(process_config).to(device)

# init contrastive loss
contrastive_loss = ContrastiveLoss(loss_type=train_config["contrastive"]["loss_type"])
mse_loss = nn.MSELoss()

# init mse loss
mse_loss = nn.MSELoss()

# get optimizers
embedder_decoder_optimizer = torch.optim.Adam(
    chain(embedder.parameters(), decoder.parameters()),
    lr=train_config["optimize"]["lr"],
    weight_decay=train_config["optimize"]["weight_decay"],
    betas=train_config["optimize"]["betas"],
    eps=train_config["optimize"]["eps"],
)

if train_config["adv"]:
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_config["optimize"]["lr"],
        weight_decay=train_config["optimize"]["weight_decay"],
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
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
best_val_acc = None
if args.ckpt_path:
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)

    embedder.load_state_dict(checkpoint["embedder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if train_config["adv"] and checkpoint["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    embedder_decoder_optimizer.load_state_dict(
        checkpoint["embedder_decoder_optimizer_state_dict"]
    )
    if (
        train_config["adv"]
        and checkpoint["discriminator_optimizer_state_dict"] is not None
    ):
        discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )

    # extract start training epoch and batch
    start_epoch = checkpoint["epoch"]
    start_batch = checkpoint.get("batch_idx", 0)

    # extract val data
    best_val_acc = checkpoint.get("average_acc", None)

if start_batch == (len(train_ds) // batch_size) - 1:
    start_batch = 0
    start_epoch += 1

# run validation after 20 back passes
run_validation = 20 * train_config["optimize"]["grad_acc_step"]

# early stopping-like scheduler
no_improve_counter = 0

# start training
for epoch in range(start_epoch, train_config["iter"]["epoch"] + 1):
    # set params for tracking
    total_train_loss = 0
    total_train_num = 0
    total_bits = 0

    # accuracy params
    total_acc_identity = 0
    total_acc_distorted = 0
    total_train_stoi = 0

    # set lambdas
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_a = train_config["optimize"]["lambda_a"] if train_config["adv"] else 0
    lambda_cl = train_config["optimize"]["lambda_cl"]

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
        wav = batch["wav"].to(device, non_blocking=True)
        curr_bs = wav.shape[0]
        msg = np.random.choice([0, 1], [curr_bs, 1, msg_length])
        msg = torch.from_numpy(msg).float() * 2 - 1
        msg = msg.to(device, non_blocking=True)

        # get the embedded audio, carrier watermarked audio and decoded message
        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        # compute stori score
        total_train_stoi += stoi_score_batch(
            wav.squeeze(1),
            embedded.squeeze(1),
            fs=process_config["audio"]["sample_rate"],
        )

        # watermark embedding loss
        wm_embedding_loss = mse_loss(embedded, wav)

        # get contrastive loss
        aug_view_1, aug_view_2 = batch["augmented_views"]
        aug_view_1 = aug_view_1.to(device, non_blocking=True)
        aug_view_2 = aug_view_2.to(device, non_blocking=True)
        feat_view_1 = decoder.get_features(aug_view_1).to(device)
        feat_view_2 = decoder.get_features(aug_view_2).to(device)
        cl_loss = contrastive_loss(feat_view_1.squeeze(1), feat_view_2.squeeze(1))

        # message loss
        decoder_msg_distorted, decoder_msg_identity = decoded
        message_loss = mse_loss(decoder_msg_distorted, msg) + mse_loss(
            decoder_msg_identity, msg
        )

        # compute accuracies
        total_acc_identity += (decoder_msg_identity >= 0).eq(msg >= 0).sum().float()
        total_acc_distorted += (decoder_msg_distorted >= 0).eq(msg >= 0).sum().float()
        total_bits += msg.numel()

        # set adversarial loss to zero
        embedder_adv_loss = 0

        # discriminator loss - first classify the embedded as true
        if train_config["adv"]:
            labels_real = torch.full((curr_bs, 1), 1, device=device).float()
            discriminator_output_embedded = discriminator(embedded)

            # get adversarial loss
            embedder_adv_loss = F.binary_cross_entropy_with_logits(
                discriminator_output_embedded, labels_real
            )

        # backward pass
        sum_loss = (
            lambda_e * wm_embedding_loss
            + lambda_m * message_loss
            + lambda_a * embedder_adv_loss
            + lambda_cl * cl_loss
        )
        print(
            f"Losses: wm_embed: {wm_embedding_loss.item():.4f}, msg: {message_loss.item():.4f}, adv: {embedder_adv_loss if train_config['adv'] else 0:.4f}, cl: {cl_loss.item():.4f}"
        )
        sum_loss.backward()

        # perform gradient accumulation
        if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
            embedder_decoder_optimizer.step()
            embedder_decoder_optimizer.zero_grad()
            if train_config["adv"]:
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()

        # update total loss and total num
        total_train_loss += sum_loss.item() * curr_bs
        total_train_num += curr_bs

        # backward pass on discriminator
        if train_config["adv"]:
            labels_real = torch.full((curr_bs, 1), 1, device=device).float()
            labels_fake = torch.full((curr_bs, 1), 0, device=device).float()
            discriminator_output_wav = discriminator(wav)
            discriminator_output_embedded = discriminator(embedded.detach())

            # get adversarial loss on real audio
            discriminator_adv_loss_wav = F.binary_cross_entropy_with_logits(
                discriminator_output_wav, labels_real
            )
            discriminator_adv_loss_wav.backward()

            # get adversarial loss on embedded and perform step
            discriminator_adv_loss_embedded = F.binary_cross_entropy_with_logits(
                discriminator_output_embedded, labels_fake
            )
            discriminator_adv_loss_embedded.backward()

            # accumulate gradients
            if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()

        tbar.set_postfix(
            loss=f"{sum_loss.item():.4f}",
            average_loss=f"{total_train_loss / total_train_num:.4f}",
            average_stoi=f"{total_train_stoi / total_train_num:.4f}",
            acc_identity=f"{total_acc_identity / total_bits:.4f}",
            acc_distorted=f"{total_acc_distorted / total_bits:.4f}",
        )

    # val step
    with torch.no_grad():

        # set all models to eval
        embedder.eval()
        decoder.eval()
        discriminator.eval()

        # set params for tracking
        total_val_loss = 0
        total_val_num = 0
        total_acc = 0
        total_bits = 0
        total_val_stoi = 0

        # set pbar for progress tracking
        vbar = tqdm(
            val_dl,
            total=len(val_dl),
            desc=f"Epoch {epoch+1} [Val]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in vbar:
            # get current audio and watermark message
            wav = batch["wav"].to(device, non_blocking=True)
            curr_bs = wav.shape[0]
            msg = np.random.choice([0, 1], [curr_bs, 1, msg_length])
            msg = torch.from_numpy(msg).float() * 2 - 1
            msg = msg.to(device, non_blocking=True)

            # get the embedded audio, carrier watermarked audio and decoded message
            embedded, carrier_wateramrked = embedder(wav, msg)
            decoded = decoder(embedded)

            # compute stoi score
            total_val_stoi += stoi_score_batch(
                wav.squeeze(1),
                embedded.squeeze(1),
                fs=process_config["audio"]["sample_rate"],
            )

            # watermark embedding loss
            wm_embedding_loss = mse_loss(embedded, wav)

            # message loss
            message_loss = mse_loss(decoded, msg)

            # get contrastive loss
            aug_view_1, aug_view_2 = batch["augmented_views"]
            aug_view_1 = aug_view_1.to(device, non_blocking=True)
            aug_view_2 = aug_view_2.to(device, non_blocking=True)
            feat_view_1 = decoder.get_features(aug_view_1).to(device)
            feat_view_2 = decoder.get_features(aug_view_2).to(device)
            cl_loss = contrastive_loss(feat_view_1.squeeze(1), feat_view_2.squeeze(1))

            # set adversarial loss to zero
            embedder_adv_loss = 0

            # discriminator loss - first classify the embedded as true
            if train_config["adv"]:
                labels_real = torch.full((curr_bs, 1), 1, device=device).float()
                discriminator_output_embedded = discriminator(embedded.detach())

                # get adversarial loss
                embedder_adv_loss = F.binary_cross_entropy_with_logits(
                    discriminator_output_embedded, labels_real
                )

            # sum loss
            sum_loss = (
                lambda_e * wm_embedding_loss
                + lambda_m * message_loss
                + lambda_a * embedder_adv_loss
                + lambda_cl * cl_loss
            )
            total_val_loss += sum_loss.item() * curr_bs
            total_val_num += curr_bs

            # measure accuracy on val dataset
            curr_acc = (decoded >= 0).eq(msg >= 0).sum().float()
            total_acc += curr_acc.item()
            total_bits += msg.numel()

            # set pbar desc
            vbar.set_postfix(
                loss=f"{sum_loss.item():.4f}",
                acc=f"{total_acc / total_bits:.3f}",
                average_loss=f"{total_val_loss / total_val_num:.4f}",
                average_stoi=f"{total_val_stoi / total_val_num:.4f}",
            )

    # Save model checkpoint
    if args.save_ckpt and (
        best_val_acc is None or (total_acc / total_bits) > best_val_acc
    ):
        best_val_acc = total_acc / total_bits
        checkpoint_path = os.path.join(
            checkpoint_dir, f"wm_model_part_dataset_cl_val_epoch_{epoch}.pt"
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
                "embedder_decoder_optimizer_state_dict": embedder_decoder_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": (
                    discriminator_optimizer.state_dict()
                    if train_config["adv"]
                    else None
                ),
                "average_train_loss": total_train_loss / total_train_num,
                "average_val_loss": total_val_loss / total_val_num,
                "average_acc": total_acc / total_bits,
            },
            checkpoint_path,
        )
        print(
            f"Checkpoint saved: {checkpoint_path} with new best average accuracy: {total_acc / total_bits}"
        )
        no_improve_counter = 0
        continue

    if best_val_acc is not None and (total_acc / total_bits) <= best_val_acc:
        no_improve_counter += 1
        print(f"No improvement in val accuracy for {no_improve_counter} validations")
        continue

    if no_improve_counter >= 3:
        for g in embedder_decoder_optimizer.param_groups:
            g["lr"] *= 0.9
        if train_config["adv"]:
            for g in discriminator_optimizer.param_groups:
                g["lr"] *= 0.9
        print(
            f"No improvement for 3 validations → reducing LR to {embedder_decoder_optimizer.param_groups[0]['lr']:.6f}"
        )
        no_improve_counter = 0
        continue


embedder.eval()
decoder.eval()

# get test set acc
total_test_acc = 0
total_test_bits = 0
for batch in tqdm(test_dl):
    # get current audio and watermark message
    wav = batch["wav"].to(device)
    curr_bs = wav.shape[0]
    msg = np.random.choice([0, 1], [curr_bs, 1, msg_length])
    msg = torch.from_numpy(msg).float() * 2 - 1
    msg = msg.to(device)

    # get the embedded audio, carrier watermarked audio and decoded message
    embedded, carrier_wateramrked = embedder(wav, msg)
    decoded = decoder(embedded)

    # measure accuracy
    acc = (decoded >= 0).eq(msg >= 0).sum().float()
    total_test_acc += acc
    total_test_bits += msg.numel()

# print results and save
print(f"Average test accuracy: {total_test_acc / total_test_bits}")
