import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import AudioDataset
from distortions.attacks import delete_samples
from model.decoder import Decoder
from model.discriminator import Discriminator
from model.embedder import Embedder

warnings.filterwarnings(
    "ignore", message=".*TorchCodec's decoder.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configs
with open("config/train_part_cl.yaml", "r") as f:
    train_config = yaml.safe_load(f)
with open("config/process.yaml", "r") as f:
    process_config = yaml.safe_load(f)
with open("config/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)


def truncate_or_pad(audio, target_len) -> np.ndarray:
    curr_len = len(audio)
    if curr_len > target_len:
        return audio[:target_len]
    if curr_len < target_len:
        return np.pad(audio, (0, target_len - curr_len))
    return audio


def evaluate(embedder, decoder, dataset, delete_ratio):
    embedder.eval()
    decoder.eval()

    total_acc = 0
    total_bits = 0
    sr = process_config["audio"]["sample_rate"]

    with torch.no_grad():
        tbar = tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"Delete Ratio {delete_ratio:.4f}",
        )
        for idx, item in tbar:
            wav = item["wav"].to(device)  # [1, T]

            # --- Step 1: random message ---
            msg = np.random.choice([0, 1], [1, 1, train_config["watermark"]["length"]])
            msg = torch.from_numpy(msg).float() * 2 - 1
            msg = msg.to(device)

            # --- Step 2: embed watermark ---
            embedded, _ = embedder(wav.unsqueeze(0), msg)  # embedded: [1, T]

            # --- Step 3: attack on embedded signal ---
            embedded_np = embedded.squeeze(0).cpu().numpy()
            flattened_embedded_np = embedded_np.flatten()
            attacked_np = truncate_or_pad(
                delete_samples(flattened_embedded_np, delete_ratio),
                process_config["audio"]["max_len"],
            )
            attacked_embedded = (
                torch.from_numpy(attacked_np).unsqueeze(0).float().to(device)
            )

            # --- Step 4: decode attacked embedded ---
            decoded = decoder(attacked_embedded.unsqueeze(0))

            # --- Step 5: bitwise accuracy ---
            acc = (decoded >= 0).eq(msg >= 0).sum().float()
            total_acc += acc.item()
            total_bits += msg.numel()

            avg_acc = total_acc / total_bits
            tbar.set_postfix(acc=f"{avg_acc:.4f}")

    return {"acc": total_acc / total_bits if total_bits > 0 else 0.0}


# --- Load models ---
embedding_dim = model_config["dim"]["embedding"]
nlayers_encoder = model_config["layer"]["nlayers_encoder"]
nlayers_decoder = model_config["layer"]["nlayers_decoder"]
msg_length = train_config["watermark"]["length"]
win_dim = model_config["audio"]["win_dim"]

# wm model
embedder = Embedder(
    process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder
)
decoder = Decoder(
    process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder
)
discriminator = Discriminator(process_config)

# wm cl model
embedder_cl = Embedder(
    process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder
)
decoder_cl = Decoder(
    process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder
)
discriminator_cl = Discriminator(process_config)

# load checkpoints
model_ckpt = "model_ckpts/wm_model_part_dataset_val_epoch_0.pt"
model_ckpt_cl = "model_ckpts/wm_model_part_dataset_cl_val_epoch_1.pt"

checkpoint = torch.load(model_ckpt, map_location=device)
embedder.load_state_dict(checkpoint["embedder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

checkpoint_cl = torch.load(model_ckpt_cl, map_location=device)
embedder_cl.load_state_dict(checkpoint_cl["embedder_state_dict"])
decoder_cl.load_state_dict(checkpoint_cl["decoder_state_dict"])

# --- Load dataset ---
test_ds = AudioDataset(process_config, split="test", take_num=200)

# --- Evaluate for delete ratios ---
delete_samples_values = [t * 0.1 for t in range(0, 11)]  # 0% to 100%
results = {}

for delete_ratio in delete_samples_values:
    # wm models on GPU, cl on CPU
    embedder.to(device)
    decoder.to(device)
    embedder_cl.to("cpu")
    decoder_cl.to("cpu")

    print(f"Evaluating delete ratio: {delete_ratio:.2f}")
    wm_metrics = evaluate(embedder, decoder, test_ds, delete_ratio)
    results[f"wm_delete_{delete_ratio:.2f}"] = wm_metrics

    # cl models on GPU, wm on CPU
    embedder.to("cpu")
    decoder.to("cpu")
    embedder_cl.to(device)
    decoder_cl.to(device)

    print(f"Evaluating delete ratio (CL): {delete_ratio:.2f}")
    cl_metrics = evaluate(embedder_cl, decoder_cl, test_ds, delete_ratio)
    results[f"cl_delete_{delete_ratio:.2f}"] = cl_metrics

# --- Extract metrics for plotting ---
wm_acc = [results[f"wm_delete_{r:.2f}"]["acc"] for r in delete_samples_values]
cl_acc = [results[f"cl_delete_{r:.2f}"]["acc"] for r in delete_samples_values]

# --- Plot results ---
plt.figure(figsize=(10, 7))

plt.plot(delete_samples_values, wm_acc, label="WM Accuracy", marker="o")
plt.plot(delete_samples_values, cl_acc, label="CL Accuracy", marker="x")

plt.xlabel("Delete Sample Ratio")
plt.ylabel("Bitwise Accuracy")
plt.title("Watermark Accuracy vs Delete Sample Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
output_dir = "results_images"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "accuracy_vs_delete_rate.png"), dpi=300)

plt.show()
