import os
import random
import warnings

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from data.dataset import AudioDataset
from distortions.attacks import delete_samples, resample
from model.utils import (
    accuracy,
    get_model_average_pesq_dataset,
    get_model_average_stoi_dataset,
    load_from_ckpt,
    truncate_or_pad_np,
)

warnings.filterwarnings(
    "ignore", message=".*TorchCodec's decoder.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

device = torch.device("cpu")

# Load configs
with open("config/train_part_cl.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)


def evaluate(embedder, decoder, dataset, resample_ratio):
    embedder.eval()
    decoder.eval()

    total_acc = 0
    total_bits = 0

    with torch.no_grad():
        tbar = tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"Resample Ratio {resample_ratio:.4f}",
        )
        for idx, item in tbar:
            wav = item["wav"].unsqueeze(0).to(device)
            curr_bs = wav.shape[0]
            msg = (
                torch.randint(
                    0,
                    2,
                    (curr_bs, 1, train_config["watermark"]["length"]),
                    device=device,
                ).float()
                * 2
                - 1
            ).to(device)

            # --- Step 2: embed watermark ---
            embedded, _ = embedder(wav, msg)  # embedded: [1, T]

            # --- Step 3: attack on embedded signal ---
            embedded_np = embedded.squeeze(0).cpu().numpy()
            flattened_embedded_np = embedded_np.flatten()
            attacked_np = truncate_or_pad_np(
                resample(
                    audio=flattened_embedded_np,
                    sr=process_config["audio"]["sample_rate"],
                    downsample_sr=resample_ratio,
                ),
                process_config["audio"]["max_len"],
            )
            attacked_embedded = (
                torch.from_numpy(attacked_np).unsqueeze(0).float().to(device)
            )

            # --- Step 4: decode attacked embedded ---
            decoded = decoder(attacked_embedded.unsqueeze(0))

            # --- Step 5: bitwise accuracy ---
            total_acc += accuracy(decoded=decoded, msg=msg)
            total_bits += msg.numel()

            avg_acc = total_acc / total_bits
            tbar.set_postfix(acc=f"{avg_acc:.4f}")

    return {"acc": total_acc / total_bits if total_bits > 0 else 0.0}


# --- Load dataset ---
test_ds = AudioDataset(process_config, split="test", take_num=100)

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
    avg_pesq = get_model_average_pesq_dataset(
        embedder=embedder,
        dataset=test_ds,
        msg_length=train_config["watermark"]["length"],
        device=device,
        sr=process_config["audio"]["sample_rate"],
    )
    avg_stoi = get_model_average_stoi_dataset(
        embedder=embedder,
        dataset=test_ds,
        msg_length=train_config["watermark"]["length"],
        device=device,
        sr=process_config["audio"]["sample_rate"],
    )
    print(
        f"Loaded model from: {model['model_ckpt']} with avg pesq: {avg_pesq} and avg stoi: {avg_stoi}"
    )
    model.update(
        {
            "embedder": embedder,
            "decoder": decoder,
            "avg_pesq": avg_pesq,
            "avg_stoi": avg_stoi,
        }
    )

# --- Pick one random audio from dataset ---
idx = random.randint(0, len(test_ds) - 1)
item = test_ds[idx]
wav = item["wav"].unsqueeze(0).to(device)  # [1, T]
orig_np = wav.squeeze(0).cpu().numpy()

# --- Evaluate for delete ratios ---
resample_values = [
    44100,  # baseline
    22050,  # half
    14700,  # one-third
    11025,  # quarter
    8820,  # one-fifth
    7350,  # one-sixth
    6300,  # one-seventh
]
results = {}

for model in config:
    embedder = model["embedder"]
    decoder = model["decoder"]
    model_desc = model["model_desc"]

    results[model_desc] = {}

    for resample_ratio in resample_values:
        print(f"Evaluating {model_desc} at resample ratio: {resample_ratio:.2f}")
        metrics = evaluate(embedder, decoder, test_ds, resample_ratio)
        results[model_desc][resample_ratio] = metrics

# --- Extract metrics for plotting ---
plt.figure(figsize=(10, 7))

for model in config:
    model_desc = model["model_desc"]
    avg_pesq = model["avg_pesq"]
    avg_stoi = model["avg_stoi"]

    acc_values = [results[model_desc][r]["acc"] for r in resample_values]
    plt.plot(
        resample_values,
        acc_values,
        marker="o",
        label=f"{model_desc} (PESQ={avg_pesq:.2f}, STOI={avg_stoi:.2f})",
    )

plt.xlabel("Downsample Rate (Hz)")
plt.ylabel("Bitwise Accuracy")
plt.title("Watermark Accuracy vs Resampling Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
output_dir = "results_images"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "accuracy_vs_resample_rate.png"), dpi=300)

plt.show()
