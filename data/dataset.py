import os

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, process_config, split="train", batch_size=100, dataset_path_prefix=""):
        self.dataset_path = dataset_path_prefix + "cv-corpus-22.0-delta-2025-06-20-en/cv-corpus-22.0-delta-2025-06-20/en/clips"
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]  # patch size
        self.max_len = process_config["audio"]["max_len"]  # in samples
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.set_train_val_data()

    def set_train_val_data(self):
        files = sorted(os.listdir(self.dataset_path))
        train_end = int(len(files) * 0.8)
        val_end = int(len(files) * 0.9)
        if self.split == "train":
            self.files = files[:train_end]
            return None
        if self.split == "val":
            self.files = files[train_end: val_end]
            return None
        self.files = files[val_end:]
        return None

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.files[idx])
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

        wav_matrix = wav.reshape(max_patch_num, self.win_len)

        sample = {
            "matrix": wav_matrix,
            "wav": wav,
            "sample_rate": sr,
            "patch_num": max_patch_num,
            "pad_num": pad_num,
            "name": self.files[idx]
        }
        return sample

    def __len__(self):
        return len(self.files)