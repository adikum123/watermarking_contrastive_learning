import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from pesq import pesq  # pip install pesq

from data.lj_dataset import LjAudioDataset
from model.utils import load_from_ckpt

# Load configs
with open("config/train_part_cl.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process_lj.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

print(process_config)

device = torch.device("cpu")
target_sr = int(process_config["audio"]["sample_rate"])

# --- Model config ---
config = [
    {
        "model_ckpt": "model_ckpts/lj/wm_model_lj_pesq_3.56_acc_0.99_dist_acc_0.86_epoch_5_gs.pt",
        "model_desc": "wm model",
    },
    {
        "model_ckpt": "model_ckpts/lj/cl/wm_model_lj_pesq_3.65_acc_1.00_dist_acc_0.84_epoch_2_gs.pt",
        "model_desc": "wm cl model",
    },
]

# Load models
for model in config:
    embedder, decoder = load_from_ckpt(
        ckpt=model["model_ckpt"],
        model_config=model_config,
        train_config=train_config,
        process_config=process_config,
        device=device,
    )
    model.update(
        {
            "embedder": embedder,
            "decoder": decoder,
        }
    )

# --- Load dataset ---
test_ds = LjAudioDataset(process_config, split="test")

# --- Pick one random audio from dataset ---
idx = random.randint(0, len(test_ds) - 1)
item = test_ds[idx]
wav = item["wav"].unsqueeze(0).to(device)  # [1, T]
orig_np = wav.squeeze(0).cpu().numpy().astype("float32").flatten()

# --- Normalize original waveform ---
orig_np = orig_np / max(1e-9, np.max(np.abs(orig_np)))  # avoid division by zero

# Save original audio
output_dir = "results_audios"
os.makedirs(output_dir, exist_ok=True)
sf.write(
    os.path.join(output_dir, f"sample_{idx}_original.wav"),
    orig_np,
    target_sr,
    format="WAV",
)

# --- For each model, embed and save ---
for model in config:
    embedder = model["embedder"]
    decoder = model["decoder"]
    model_desc = model["model_desc"]

    embedder.eval()
    decoder.eval()

    # Generate random message
    msg = (
        torch.randint(
            0,
            2,
            (1, 1, train_config["watermark"]["length"]),
            device=device,
        ).float()
        * 2
        - 1
    )

    # Embed watermark
    with torch.no_grad():
        embedded, _ = embedder(wav, msg)  # [1, T]

    embedded_np = embedded.squeeze(0).cpu().numpy().astype("float32").flatten()

    # --- Normalize embedded waveform ---
    embedded_np = embedded_np / max(1e-9, np.max(np.abs(embedded_np)))

    # --- Compute PESQ (resample to 16 kHz) ---
    try:
        orig_16k = librosa.resample(orig_np, orig_sr=target_sr, target_sr=16000)
        embedded_16k = librosa.resample(embedded_np, orig_sr=target_sr, target_sr=16000)
        pesq_score = pesq(16000, orig_16k, embedded_16k, 'wb')  # wideband
    except Exception as e:
        print(f"Warning: PESQ computation failed: {e}")
        pesq_score = -1  # fallback

    # Debug info
    print(
        f"[DEBUG] Embedded audio ({model_desc}) shape: {embedded_np.shape}, "
        f"dtype: {embedded_np.dtype}, PESQ: {pesq_score:.2f}"
    )

    # Save embedded audio with PESQ in filename
    sf.write(
        os.path.join(
            output_dir,
            f"sample_{idx}_{model_desc.replace(' ', '_')}_pesq_{pesq_score:.2f}.wav",
        ),
        embedded_np,
        target_sr,
        format="WAV",
    )

print(f"✅ Saved original and embedded versions of sample {idx} to {output_dir}")
