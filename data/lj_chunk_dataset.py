import os

import torch
from torch.utils.data import Dataset


class PrecomputedLjAudioDataset(Dataset):
    """
    Loads precomputed concatenated batches of audio samples.
    Each .pt file contains {"wav": Tensor[N, 1, T], "name": List[str]}.
    Exposes individual samples to DataLoader for flexible batching.
    """

    def __init__(self, split="train", base_dir="/opt/dlami/nvme/lj_speech_processed"):
        chunk_dir = os.path.join(base_dir, split)
        self.chunk_files = sorted(
            [
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.endswith(".pt")
            ]
        )
        if not self.chunk_files:
            raise RuntimeError(f"No .pt chunks found in {chunk_dir}")

        # build index mapping
        self.index_map = []
        self.chunk_sizes = []
        for ci, chunk_path in enumerate(self.chunk_files):
            meta = torch.load(chunk_path, map_location="cpu")
            n_samples = meta["wav"].shape[0]
            self.chunk_sizes.append(n_samples)
            self.index_map.extend([(ci, si) for si in range(n_samples)])

        self._cache = {}

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_idx, sample_idx = self.index_map[idx]

        if chunk_idx not in self._cache:
            self._cache[chunk_idx] = torch.load(
                self.chunk_files[chunk_idx], map_location="cpu"
            )

        chunk = self._cache[chunk_idx]

        wav = chunk["wav"][sample_idx]  # shape [1, T]
        name = chunk["name"][sample_idx]
        return {"wav": wav, "name": name}
