import os

import torch
import yaml
from torch.utils.data import DataLoader

from model.utils import get_datasets

with open("config/process_lj.yaml", "r") as f:
    process_config = yaml.safe_load(f)

# --- your dataset setup ---
batch_size = 4
train_ds, val_ds, test_ds = get_datasets(
    contrastive=False,
    process_config=process_config,
    take_part=False,
    dataset_type="ljspeech",
)


# --- dataloader setup ---
def make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=None,
        timeout=0,
    )


train_dl = make_loader(train_ds, shuffle=True)
val_dl = make_loader(val_ds, shuffle=False)
test_dl = make_loader(test_ds, shuffle=False)


# --- preprocessing function ---
def precompute_and_save(
    dataloader,
    split,
    out_root="/opt/dlami/nvme/lj_speech_processed",
    batches_per_chunk=25,
):
    """
    Precompute audio batches and save them as .pt chunks
    """
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    buffer = {"wav": [], "name": []}
    chunk_idx = 0

    for i, batch in enumerate(dataloader):
        buffer["wav"].append(batch["wav"])  # shape [B, 1, T]
        buffer["name"].extend(batch["name"])

        # Once we reach N batches, save and reset
        if (i + 1) % batches_per_chunk == 0:
            wav_cat = torch.cat(buffer["wav"], dim=0)  # shape [N*B, 1, T]
            chunk = {"wav": wav_cat, "name": buffer["name"]}
            out_path = os.path.join(out_dir, f"chunk_{chunk_idx:05d}.pt")
            torch.save(chunk, out_path)
            print(f"✅ Saved {out_path} with {wav_cat.shape[0]} samples")

            buffer = {"wav": [], "name": []}
            chunk_idx += 1

    # Save any leftover samples
    if buffer["wav"]:
        wav_cat = torch.cat(buffer["wav"], dim=0)
        chunk = {"wav": wav_cat, "name": buffer["name"]}
        out_path = os.path.join(out_dir, f"chunk_{chunk_idx:05d}.pt")
        torch.save(chunk, out_path)
        print(
            f"✅ Saved {out_path} with {wav_cat.shape[0]} samples (last partial chunk)"
        )


# --- run for each split ---
precompute_and_save(train_dl, "train")
precompute_and_save(val_dl, "val")
precompute_and_save(test_dl, "test")
