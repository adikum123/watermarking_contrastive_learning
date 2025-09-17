import os
import random

import soundfile as sf
import torch
import yaml

from data.dataset import AudioDataset
from model.utils import load_from_ckpt

# Load configs
with open("config/train_part_cl.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

device = torch.device("cpu")

# config
config = [
    {
        "model_ckpt": "model_ckpts/finetune_wm_model_pesq_3.359905179619789_acc_0.9998.pt",
        "model_desc": "wm model",
    },
    {
        "model_ckpt": "model_ckpts/wm_model_cl_pesq_1.753751079440117_acc_1.0.pt",
        "model_desc": "wm cl model",
    },
]

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
test_ds = AudioDataset(process_config, split="test", take_num=100)

# --- Pick one random audio from dataset ---
idx = random.randint(0, len(test_ds) - 1)
item = test_ds[idx]
wav = item["wav"].unsqueeze(0).to(device)  # [1, T]
orig_np = wav.squeeze(0).cpu().numpy().astype("float32").flatten()

# Debug info
print(f"[DEBUG] Picked sample index: {idx}")
print(f"[DEBUG] Original audio shape: {orig_np.shape}, dtype: {orig_np.dtype}")

# Save original audio
output_dir = "results_audios"
os.makedirs(output_dir, exist_ok=True)
sf.write(
    os.path.join(output_dir, f"sample_{idx}_original.wav"),
    orig_np,
    int(process_config["audio"]["sample_rate"]),
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

    # Debug info
    print(
        f"[DEBUG] Embedded audio ({model_desc}) shape: {embedded_np.shape}, dtype: {embedded_np.dtype}"
    )

    # Save embedded audio
    sf.write(
        os.path.join(
            output_dir, f"sample_{idx}_{model_desc.replace(' ', '_')}_embedded.wav"
        ),
        embedded_np,
        int(process_config["audio"]["sample_rate"]),
        format="WAV",
    )

print(f"✅ Saved original and embedded versions of sample {idx} to {output_dir}")
