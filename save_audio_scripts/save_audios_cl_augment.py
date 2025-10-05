import os
import random
import time

import soundfile as sf
import torch
import yaml

from data.lj_dataset import LjAudioDataset
from distortions.cl_augmentations import ContrastiveAugmentations
from model.utils import truncate_or_pad_np

# Load configs
with open("config/process_lj.yaml", "r") as f:
    process_config = yaml.safe_load(f)

print(process_config)

device = torch.device("cpu")
target_sr = int(process_config["audio"]["sample_rate"])

# --- Load dataset ---
test_ds = LjAudioDataset(process_config, split="test")

# --- Init contrastive augment ----
cl_aug = ContrastiveAugmentations(sr=target_sr)

# --- Pick one random audio from dataset ---
idx = random.randint(0, len(test_ds) - 1)
item = test_ds[idx]
wav = item["wav"].unsqueeze(0).to(device)  # [1, T]
orig_np = wav.squeeze(0).cpu().numpy().astype("float32").flatten()

# Save original audio
output_dir = "results_audios"
os.makedirs(output_dir, exist_ok=True)
sf.write(
    os.path.join(output_dir, f"sample_{idx}_original.wav"),
    orig_np,
    target_sr,
    format="WAV",
)

# --- Apply contrastive augmentations ---
for iter_idx in range(3):
    start = time.time()
    cl_aug_audio_np = truncate_or_pad_np(cl_aug.apply(audio=orig_np), target_len=process_config["audio"]["max_len"])
    end = time.time()
    print(f"Applied contrastive augmentations in {end - start:.2f} seconds")
    sf.write(
        os.path.join(output_dir, f"sample_{idx}_cl_aug_{iter_idx}.wav"),
        cl_aug_audio_np,
        target_sr,
        format="WAV",
    )
