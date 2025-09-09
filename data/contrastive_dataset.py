import os
import csv

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from distortions.attacks import (delete_samples, mp3_compression,
                                 pcm_bit_depth_conversion, resample)


class ContrastiveAudioDataset(Dataset):

    def __init__(self, process_config, split="train", take_num=None):
        self.dataset_path = os.path.join("mnt", "s3", "data", "raw", "clips")
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]  # patch size
        self.max_len = process_config["audio"]["max_len"]  # in samples
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.take_num = take_num
        self.set_data()

    def set_data(self):
        file_name = "train" if self.split == "train" else "dev" if self.split == "val" else "test"
        file_path = os.path.join("mnt", "s3", "data", "raw", f"{file_name}.tsv")
        self.files = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.files.append(dict(row))
        if self.take_num:
            # sort first by path, then take first self.take_num items
            self.files = sorted(self.files, key=lambda x: x["path"])[:self.take_num]

    def truncate_or_pad(self, wav):
        # get actual length of the audio
        actual_length = wav.shape[1]
        max_patch_num = self.max_len // self.win_len
        target_len = max_patch_num * self.win_len

        # Truncate the audio to max_len if longer
        if wav.shape[1] > target_len:
            return wav[:, :target_len]

        # Otherwise, pad zeros to reach max_len
        if wav.shape[1] < target_len:
            pad_num = target_len - wav.shape[1]
            return torch.cat((wav, torch.zeros(1, pad_num)), dim=1)

        return wav

    def generate_augmented_views(self, x):
        """
        Generate two random augmented views of audio sample `x` using attack.py
        """
        def random_augment(audio, sr):
            aug_audio = np.copy(audio)

            if np.random.rand() < 0.5:
                aug_audio = mp3_compression(aug_audio, sr, quality=np.random.choice([2, 4, 6]))

            if np.random.rand() < 0.5:
                bit_depth = np.random.choice([8, 16, 24])
                aug_audio = pcm_bit_depth_conversion(aug_audio, sr, pcm=bit_depth)

            if np.random.rand() < 0.5:
                deletion_percentage = np.random.uniform(0.1, 0.5)
                if len(aug_audio) > int(sr * deletion_percentage):
                    aug_audio = delete_samples(aug_audio, sr, deletion_percentage)

            if np.random.rand() < 0.5:
                aug_audio = resample(aug_audio, sr)

            return aug_audio

        # Remove channel dim (1, N) → (N,)
        x_np = x.squeeze().numpy()

        # Apply augmentations
        view1 = random_augment(x_np, self.sample_rate)
        view2 = random_augment(x_np, self.sample_rate)

        # Restore shape (N,) → (1, N)
        return (
            self.truncate_or_pad(torch.from_numpy(view1).unsqueeze(0).float()),
            self.truncate_or_pad(torch.from_numpy(view2).unsqueeze(0).float()),
        )


    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.files[idx]["path"])
        wav, sr = torchaudio.load(file_path, format="mp3")
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
            sr = self.sample_rate

        # sample
        sample = {
            "wav": self.truncate_or_pad(wav),
            "augmented_views": self.generate_augmented_views(wav),
            "name": self.files[idx],
        }
        return sample

    def __len__(self):
        return len(self.files)