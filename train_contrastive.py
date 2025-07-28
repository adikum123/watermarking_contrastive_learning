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
from model.detector import Decoder
from model.discriminator import Discriminator
from model.embedder import Embedder

# Load config
with open("config/train_contrastive.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# DataLoader setup
batch_size = train_config["optimize"]["batch_size"]
train_ds = ContrastiveAudioDataset(process_config, split="train")
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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

for epoch in range(train_config["iter"]["epoch"] + 1):
    # set params for tracking
    total_loss = 0
    total_num = 0

    # set lambdas
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_a = train_config["optimize"]["lambda_a"] if train_config["adv"] else 0
    lambda_cl = train_config["optimize"]["lambda_cl"]
    for batch in tqdm(train_dl):
        # get current audio and watermark message
        wav = batch["wav"].to(device)
        msg = np.random.choice([0,1], [batch_size, 1, msg_length])
        msg = torch.from_numpy(msg).float()*2 - 1

        # set zero grad
        embedder_decoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # get the embedded audio, carrier watermarked audio and decoded message
        embedded, carrier_wateramrked = embedder(wav, msg)
        decoded = decoder(embedded)

        # watermark embedding loss
        wm_embedding_loss = mse_loss(embedded, wav)

        # message loss
        decoder_msg_distorted, _, decoder_msg_identity, _ = decoded
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

        print(f"Sum loss for current batch: {sum_loss.item()}")

        # perform step
        embedder_decoder_optimizer.step()

        # backward pass on discriminator
        if train_config["adv"]:
            labels_real = torch.full((batch_size, 1), 1, device=device).float()
            labels_fake = torch.full((batch_size, 1), 0, device=device).float()

            # classify encoded audio as fake and real audio as real
            discriminator_output_embedded = discriminator(embedded)
            discriminator_output_real = discriminator(wav)

            # get adversarial loss and perform backwargs pass
            d_adv_loss = F.binary_cross_entropy_with_logits(discriminator_output_embedded, labels_fake) + bce_loss(discriminator_output_real, labels_real)
            d_adv_loss.backward()

            # perform step
            discriminator_optimizer.step()




