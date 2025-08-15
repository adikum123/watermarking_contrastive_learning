import argparse
import json
import os
from itertools import chain

import matplotlib.pyplot as plt
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

# Load config
with open("config/train_contrastive.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

parser = argparse.ArgumentParser(
    description="Train audio watermarking model with contrastive learning"
)
parser.add_argument(
    "--dataset_path_prefix", type=str, default="", help="Dataset path prefix when run from colab bro"
)
parser.add_argument(
    "--save_ckpt", action="store_true", help="Store model ckpts on google drive"
)
parser.add_argument(
    "--ckpt_path", type=str, default="", help="Path to checkpoint to resume training"
)
args = parser.parse_args()

# DataLoader setup
batch_size = train_config["optimize"]["batch_size"]
train_ds = ContrastiveAudioDataset(process_config, split="train", dataset_path_prefix=args.dataset_path_prefix)
val_ds = ContrastiveAudioDataset(process_config, split="val", dataset_path_prefix=args.dataset_path_prefix)
test_ds = ContrastiveAudioDataset(process_config, split="test", dataset_path_prefix=args.dataset_path_prefix)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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
embedder = Embedder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder).to(device)
decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder).to(device)
discriminator = Discriminator(process_config).to(device)

# init contrastive loss
contrastive_loss = ContrastiveLoss(loss_type=train_config["contrastive"]["loss_type"])
mse_loss = nn.MSELoss()

# get optimizers
embedder_decoder_optimizer = torch.optim.Adam(
    chain(embedder.parameters(), decoder.parameters()),
    lr=train_config["optimize"]["lr"],
    weight_decay=train_config["optimize"]["weight_decay"],
    betas=train_config["optimize"]["betas"],
    eps=train_config["optimize"]["eps"],
)
embedder_decoder_scheduler = torch.optim.lr_scheduler.StepLR(
    embedder_decoder_optimizer,
    step_size=train_config["optimize"]["step_size"],
    gamma=train_config["optimize"]["gamma"],
)

if train_config["adv"]:
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_config["optimize"]["lr"],
        weight_decay=train_config["optimize"]["weight_decay"],
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
    )
    discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
        discriminator_optimizer,
        step_size=train_config["optimize"]["step_size"],
        gamma=train_config["optimize"]["gamma"]
    )

print(f"Training with params:\n{json.dumps(train_config, indent=4)}\nLength of train dataset: {len(train_ds)}")

# init checkpoint dir
checkpoint_dir = "/content/drive/MyDrive/audio_watermark_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# load from checkpoint if it exists
start_epoch = 0
if args.ckpt_path:
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)

    embedder.load_state_dict(checkpoint["embedder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if train_config["adv"] and checkpoint["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    embedder_decoder_optimizer.load_state_dict(checkpoint["embedder_decoder_optimizer_state_dict"])
    if train_config["adv"] and checkpoint["discriminator_optimizer_state_dict"] is not None:
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

best_val_acc = None

# start training
for epoch in range(train_config["iter"]["epoch"] + 1):
    # set params for tracking
    total_loss = 0
    total_train_num = 0

    # set lambdas
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_a = train_config["optimize"]["lambda_a"] if train_config["adv"] else 0
    lambda_cl = train_config["optimize"]["lambda_cl"]

    # set all models to train mode
    embedder.train()
    decoder.train()
    discriminator.train()

    # set pbar for progress
    pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1} [Train]")
    for i, batch in pbar:
        # get current audio and watermark message
        wav = batch["wav"].to(device)
        msg = np.random.choice([0,1], [batch_size, 1, msg_length])
        msg = torch.from_numpy(msg).float() * 2 - 1
        msg = msg.to(device)

        # get the embedded audio, carrier watermarked audio and decoded message
        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        # watermark embedding loss
        wm_embedding_loss = mse_loss(embedded, wav)

        # message loss
        decoder_msg_distorted, decoder_msg_identity = decoded
        message_loss = mse_loss(decoder_msg_distorted, msg) + mse_loss(decoder_msg_identity, msg)

        # get contrastive loss
        aug_view_1, aug_view_2 = batch["augmented_views"]
        feat_view_1 = decoder.get_features(aug_view_1)
        feat_view_2 = decoder.get_features(aug_view_2)
        cl_loss = contrastive_loss(feat_view_1.squeeze(1), feat_view_2.squeeze(1))
        print(f"Augmented views shape: {feat_view_1.shape} {feat_view_2.shape}\nContrastive loss: {cl_loss.item()}")

        # set adversarial loss to zero
        embedder_adv_loss = 0

        # discriminator loss - first classify the embedded as true
        if train_config["adv"]:
            labels_real = torch.full((batch_size, 1), 1, device=device).float()
            discriminator_output_embedded = discriminator(embedded)

            # get adversarial loss
            embedder_adv_loss = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_real)

        # backward pass with contrastive loss
        sum_loss = lambda_e * wm_embedding_loss + lambda_m * message_loss + lambda_a * embedder_adv_loss + lambda_cl * cl_loss
        sum_loss.backward()

        # perform gradient accumulation
        if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
            embedder_decoder_optimizer.step()
            embedder_decoder_optimizer.zero_grad()
            if train_config["adv"]:
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()

        # update params
        total_train_loss += sum_loss.item()
        total_train_num += wav.shape[0]

        # backward pass on discriminator
        if train_config["adv"]:
            labels_real = torch.full((batch_size, 1), 1, device=device).float()
            labels_fake = torch.full((batch_size, 1), 0, device=device).float()
            discriminator_output_embedded = discriminator(embedded.detach())
            discriminator_output_real = discriminator(wav)

            # get adversarial loss on real audio
            discriminator_adv_loss_wav = F.binary_cross_entropy_with_logits(discriminator_output_wav, labels_real)
            discriminator_adv_loss_wav.backward()

            # get adversarial loss on embedded and perform step
            disciminator_adv_loss_embedded = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_fake)
            discriminator_adv_loss_embedded.backward()

            # accumulate gradients
            if (i + 1) % train_config["optimize"]["grad_acc_step"] == 0:
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()

            pbar.set_postfix(loss=f"{sum_loss.item():.4f}", average_loss=f"{total_train_loss / total_train_num:.4f}")

    print(f"Epoch: {epoch+1} average train loss: {total_train_loss / total_train_num}")

    # val step
    with torch.no_grad():
        # set all models to eval
        embedder.eval()
        decoder.eval()
        discriminator.eval()

        # set params for tracking
        total_val_loss = 0
        total_val_num = 0
        total_acc= 0
        total_bits = 0

        # set pbar for progress tracking
        pbar = tqdm(val_dl, total=len(val_dl), desc=f"Epoch {epoch+1} [Val]", leave=False, dynamic_ncols=True)
        for batch in pbar:
            # get current audio and watermark message
            wav = batch["wav"].to(device)
            msg = np.random.choice([0,1], [batch_size, 1, msg_length])
            msg = torch.from_numpy(msg).float() * 2 - 1
            msg = msg.to(device)

            # get the embedded audio, carrier watermarked audio and decoded message
            embedded, carrier_wateramrked = embedder(wav, msg)
            decoded = decoder(embedded)

            # watermark embedding loss
            wm_embedding_loss = mse_loss(embedded, wav)

            # message loss
            message_loss = mse_loss(decoded, msg)

            # get contrastive loss
            aug_view_1, aug_view_2 = batch["augmented_views"]
            feat_view_1 = decoder.get_features(aug_view_1)
            feat_view_2 = decoder.get_features(aug_view_2)
            cl_loss = contrastive_loss(feat_view_1.squeeze(1), feat_view_2.squeeze(1))

            # set adversarial loss to zero
            embedder_adv_loss = 0

            # discriminator loss - first classify the embedded as true
            if train_config["adv"]:
                labels_real = torch.full((batch_size, 1), 1, device=device).float()
                discriminator_output_embedded = discriminator(embedded)

                # get adversarial loss
                embedder_adv_loss = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_real)

            # compute sum loss and update params
            sum_loss = lambda_e * wm_embedding_loss + lambda_m * message_loss + lambda_a * embedder_adv_loss + lambda_cl * cl_loss
            total_val_loss += sum_loss.item() * curr_bs
            total_val_num += curr_bs

            # measure accuracy on val dataset
            curr_acc = (decoded >= 0).eq(msg >= 0).sum().float()
            total_acc += curr_acc.item()
            total_bits += msg.numel()

            # set pbar desc
            pbar.set_postfix(loss=f"{sum_loss.item():.4f}", acc=f"{total_acc / total_bits:.3f}", average_loss=f"{total_val_loss / total_val_num:.4f}")

    print(f"Epoch: {epoch+1} average val loss: {total_val_loss / total_val_num}")
    print(f"Epoch: {epoch+1} average acc: {total_acc / total_bits}")

    # Step the schedulers here (once per epoch)
    embedder_decoder_scheduler.step()
    if train_config["adv"]:
        discriminator_scheduler.step()

    # Save model checkpoint
    if args.save_ckpt and (best_val_acc is None or (total_acc / total_bits) > best_val_acc):
        best_val_acc = total_acc / total_bits
        checkpoint_path = os.path.join(checkpoint_dir, f"wm_model_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "embedder_state_dict": embedder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "discriminator_state_dict": discriminator.state_dict() if train_config["adv"] else None,
            "embedder_decoder_optimizer_state_dict": embedder_decoder_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict() if train_config["adv"] else None,
            "average_train_loss": total_train_loss / total_train_num,
            "average_val_loss": total_val_loss / total_val_num,
            "average_acc": total_acc / total_val_num
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path} with new best average accuracy: {total_acc / total_bits}")

embeddder.eval()
decoder.eval()

# get test set acc
total_test_acc = 0
total_test_num = 0
for batch in tqdm(test_dl):
    # get current audio and watermark message
    wav = batch["wav"].to(device)
    msg = np.random.choice([0,1], [batch_size, 1, msg_length])
    msg = torch.from_numpy(msg).float() * 2 - 1

    # get the embedded audio, carrier watermarked audio and decoded message
    embedded, carrier_wateramrked = embedder(wav, msg)
    decoded = decoder(embedded)

    # measure accuracy
    acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
    total_test_acc += acc
    total_test_num += wav.shape[0]

# print results and save
print(f"Average test accuracy: {total_test_acc / total_test_num}")
embedder.save()
decoder.save()




