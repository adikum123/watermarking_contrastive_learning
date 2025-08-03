import json
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import AudioDataset
from model.detector import Decoder
from model.discriminator import Discriminator
from model.embedder import Embedder

# Load config
with open("config/train.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# DataLoader setup
batch_size = train_config["optimize"]["batch_size"]
train_ds = AudioDataset(process_config, split="train")
val_ds = AudioDataset(process_config, split="val")
test_ds = AudioDataset(process_config, split="test")
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

# Models
embedder = Embedder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder).to(device)
decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder).to(device)
discriminator = Discriminator(process_config).to(device)

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

print(f"Training with params:\n{json.dumps(train_config, indent=4)}\n")


for epoch in range(train_config["iter"]["epoch"] + 1):
    # set params for tracking
    total_train_loss = 0
    total_train_num = 0

    # set lambdas
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_a = train_config["optimize"]["lambda_a"] if train_config["adv"] else 0

    # set all models to train mode
    embedder.train()
    decoder.train()
    discriminator.train()
    for batch in tqdm(train_dl):
        # get current audio and watermark message
        wav = batch["wav"].to(device)
        msg = np.random.choice([0,1], [batch_size, 1, msg_length])
        msg = torch.from_numpy(msg).float() * 2 - 1

        # set zero grad
        embedder_decoder_optimizer.zero_grad()
        if train_config["adv"]:
            discriminator_optimizer.zero_grad()

        # get the embedded audio, carrier watermarked audio and decoded message
        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        # watermark embedding loss
        wm_embedding_loss = mse_loss(embedded, wav)

        # message loss
        decoder_msg_distorted, _, decoder_msg_identity, _ = decoded
        message_loss = mse_loss(decoder_msg_distorted, msg) + mse_loss(decoder_msg_identity, msg)

        # set adversarial loss to zero
        embedder_adv_loss = 0

        # discriminator loss - first classify the embedded as true
        if train_config["adv"]:
            labels_real = torch.full((batch_size, 1), 1, device=device).float()
            discriminator_output_embedded = discriminator(embedded)

            # get adversarial loss
            embedder_adv_loss = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_real)

        # backward pass
        sum_loss = lambda_e * wm_embedding_loss + lambda_m * message_loss + lambda_a * embedder_adv_loss
        sum_loss.backward()

        # perform step
        embedder_decoder_optimizer.step()

        # update total loss and total num
        total_train_loss += sum_loss.item()
        total_train_num += wav.shape[0]

        # backward pass on discriminator
        if train_config["adv"]:
            labels_real = torch.full((batch_size, 1), 1, device=device).float()
            labels_fake = torch.full((batch_size, 1), 0, device=device).float()
            discriminator_output_wav = discriminator(wav)

            # get adversarial loss on real audio
            discriminator_adv_loss_wav = F.binary_cross_entropy_with_logits(discriminator_output_wav, labels_real)
            discriminator_optimizer.step()

            # classify encoded audio as fake and real audio as real
            discriminator_output_wav = discriminator(wav)
            discriminator_output_embedded = discriminator(embedded.detach())

            # get adversarial loss on embedded and perform step
            disciminator_adv_loss_embedded = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_fake)
            discriminator_optimizer.step()

        print(f"Curr train loss: {sum_loss.item()}")

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
        total_acc_distorted = 0
        total_acc_identiity = 0

        for batch in tqdm(val_dl):
            # get current audio and watermark message
            wav = batch["wav"].to(device)
            msg = np.random.choice([0,1], [batch_size, 1, msg_length])
            msg = torch.from_numpy(msg).float() * 2 - 1

            # get the embedded audio, carrier watermarked audio and decoded message
            embedded, carrier_wateramrked = embedder(wav, msg)
            decoded = decoder(embedded)

            # watermark embedding loss
            wm_embedding_loss = mse_loss(embedded, wav)

            # message loss
            message_loss = mse_loss(decoded, msg)

            # set adversarial loss to zero
            embedder_adv_loss = 0

            # discriminator loss - first classify the embedded as true
            if train_config["adv"]:
                labels_real = torch.full((batch_size, 1), 1, device=device).float()
                discriminator_output_embedded = discriminator(embedded)

                # get adversarial loss
                embedder_adv_loss = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_real)

            # sum loss
            sum_loss = lambda_e * wm_embedding_loss + lambda_m * message_loss + lambda_a * embedder_adv_loss
            total_val_loss += sum_loss
            total_val_num += wav.shape[0]
            print(f"Curr val loss: {sum_loss}")

            # measure accuracy on val dataset
            decoder_acc_distorted = (decoder_msg_distorted >= 0).eq(msg >= 0).sum().float() / msg.numel()
            decoder_acc_identity = (decoder_msg_identity >= 0).eq(msg >= 0).sum().float() / msg.numel()
            total_acc_distorted += decoder_acc_distorted
            total_acc_identity += decoder_acc_identity

    print(f"Epoch: {epoch+1} average val loss: {total_val_loss / total_val_num}")
    print(f"Epoch: {epoch+1} average dist acc: {total_acc_distorted / total_val_num}")
    print(f"Epoch: {epoch+1} average identity acc: {total_acc_identity / total_val_num}")

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