import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.contrastive_dataset import ContrastiveAudioDataset
from data.dataset import AudioDataset
from model.detector import Decoder
from model.discriminator import Discriminator
from model.embedder import Embedder

# Load config
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

# DataLoader setup
batch_size = 32
train_ds = ContrastiveAudioDataset(process_config, split="train")
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# Model config
embedding_dim = model_config["dim"]["embedding"]
nlayers_encoder = model_config["layer"]["nlayers_encoder"]
nlayers_decoder = model_config["layer"]["nlayers_decoder"]
msg_length = model_config["wm"]["msg_length"]
win_dim = model_config["audio"]["win_dim"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
embedder = Embedder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder).to(device)
decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder).to(device)
discriminator = Discriminator(process_config).to(device)

# Length stats collection
all_lengths = []
max_actual_length = 0

for batch in tqdm(train_dl):
    batch_lengths = batch["actual_length"]  # tensor of shape [batch_size]
    all_lengths.extend(batch_lengths.tolist())  # collect all lengths
    batch_max_len = batch_lengths.max().item()
    if batch_max_len > max_actual_length:
        max_actual_length = batch_max_len

print(f"Max actual audio length: {max_actual_length} samples")

# 📊 Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(all_lengths, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Actual Audio Lengths")
plt.xlabel("Audio Length (samples)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
