import os
import csv

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, process_config, split="train", batch_size=100, dataset_path_prefix=""):
        self.dataset_path = os.path.join("mnt", "s3", "data", "raw", "clips")
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]  # patch size
        self.max_len = process_config["audio"]["max_len"]  # in samples
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.set_data()

    def set_data(self):
        file_name = "train" if self.split == "train" else "dev" if self.split == "val" else "test"
        file_path = os.path.join("mnt", "s3", "data", "raw", f"{file_name}.tsv")
        self.files = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.files.append(dict(row))

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.files[idx]["path"])
        wav, sr = torchaudio.load(file_path, format="mp3")
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
            sr = self.sample_rate
        max_patch_num = self.max_len // self.win_len
        target_len = max_patch_num * self.win_len

        # Truncate the audio to max_len if longer
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
            pad_num = target_len - wav.shape[1] # negative padding imples truncated

        # Otherwise, pad zeros to reach max_len
        elif wav.shape[1] < target_len:
            pad_num = target_len - wav.shape[1]
            wav = torch.cat((wav, torch.zeros(1, pad_num)), dim=1)
        else:
            pad_num = 0

        sample = {
            "wav": wav,
            "sample_rate": sr,
            "patch_num": max_patch_num,
            "pad_num": pad_num,
            "name": self.files[idx]["path"]
        }
        return sample

    def __len__(self):
        return len(self.files)