import csv
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, process_config, split="train", take_num=None):
        """
        Args:
            process_config (dict): audio configs
            split (str): "train", "val", or "test"
            take_num (int): optional, limit number of samples
        """
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.take_num = take_num
        self.set_data()

    def set_data(self):
        self.dataset_path = os.path.join("mnt", "s3", "data", "raw", "clips")
        file_name = (
            "train"
            if self.split == "train"
            else "dev" if self.split == "val" else "test"
        )
        file_path = os.path.join("mnt", "s3", "data", "raw", f"{file_name}.tsv")
        self.files = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.files.append(dict(row))

        # Optional: limit number of samples
        if self.take_num:
            self.files = sorted(self.files, key=lambda x: x["path"])[: self.take_num]

    def __getitem__(self, idx):
        file_path = self.files[idx]["path"]

        # Load audio
        wav, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            wav = resampler(wav)
            sr = self.sample_rate

        target_len = self.max_len

        # Truncate
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        # Pad
        elif wav.shape[1] < target_len:
            pad_num = target_len - wav.shape[1]
            wav = torch.cat((wav, torch.zeros(1, pad_num)), dim=1)

        return {"wav": wav, "sample_rate": sr, "name": os.path.basename(file_path)}

    def __len__(self):
        return len(self.files)
