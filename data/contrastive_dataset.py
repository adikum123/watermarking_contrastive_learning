import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from distortions.attacks import (delete_samples, mp3_compression,
                                 pcm_bit_depth_conversion, resample)


class ContrastiveAudioDataset(Dataset):

    def __init__(self, process_config, split="train", batch_size=100):
        self.dataset_path = "cv-corpus-22.0-delta-2025-06-20-en/cv-corpus-22.0-delta-2025-06-20/en/clips"
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]  # patch size
        self.max_len = process_config["audio"]["max_len"]  # in samples
        self.split = split
        self.set_train_val_data()

    def set_train_val_data(self):
        files = sorted(os.listdir(self.dataset_path))
        split_index = int(len(files) * 0.8)
        if self.split == "train":
            self.files = files[:split_index]
            return None
        self.files = files[split_index:]
        return None


    def truncate_or_pad(self, wav):
        length = wav.shape[1]
        if length > self.max_len:
            wav = wav[:, :max_len]
            return wav
        if length < self.max_len:
            pad = torch.zeros(1, self.max_len - length, dtype=wav.dtype)
            wav = torch.cat([wav, pad], dim=1)
            return wav
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
                deletion_percentage = np.random.uniform(0.01, 0.1)
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
            self.truncate_or_pad(torch.from_numpy(view1).unsqueeze(0)),
            self.truncate_or_pad(torch.from_numpy(view2).unsqueeze(0)),
        )


    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.files[idx])
        wav, sr = torchaudio.load(file_path, format="mp3")
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
            sr = self.sample_rate

        # get actual length of the audio
        actual_length = wav.shape[1]

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
            "augmented_views": self.generate_augmented_views(wav),
            "sample_rate": sr,
            "patch_num": max_patch_num,
            "pad_num": pad_num,
            "name": self.files[idx],
            "actual_length": actual_length
        }
        return sample

    def __len__(self):
        return len(self.files)