from io import BytesIO
from typing import Tuple

import boto3
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from distortions.attacks import (delete_samples, mp3_compression,
                                 pcm_bit_depth_conversion, resample)


def _parse_s3_url(url: str) -> Tuple[str, str]:
    assert url.startswith("s3://"), f"expected s3://..., got {url}"
    path = url[5:]
    bucket, key_prefix = path.split("/", 1)
    return bucket, key_prefix.rstrip("/")

class LjAudioDataset(Dataset):
    def __init__(
        self, process_config, split="train", s3_root="s3://ml-deepmark/data/lj_speech_dataset/LJSpeech-1.1", contrastive=False
    ):
        self.s3_root = s3_root
        self.s3 = boto3.client("s3")
        self.bucket, self.prefix = _parse_s3_url(s3_root)

        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]

        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.contrastive = contrastive
        self.set_data()

    def set_data(self):
        # Download metadata.csv from S3
        metadata_key = f"{self.prefix}/metadata.csv"
        obj = self.s3.get_object(Bucket=self.bucket, Key=metadata_key)
        lines = obj["Body"].read().decode("utf-8").strip().split("\n")
        lines = [line.split("|") for line in lines]

        # Each entry: (wav_file, normalized_text, original_text)
        self.files = [
            {"key": f"{self.prefix}/wavs/{line[0]}.wav"}
            for line in lines
        ]

        # Simple split: ~12k train, 500 val, 500 test
        if self.split == "train":
            self.files = self.files[:12000]
        elif self.split == "val":
            self.files = self.files[12000:12500]
        else:
            self.files = self.files[12500:]

    def truncate_or_pad(self, wav):
        target_len = self.max_len

        # Truncate
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        # Pad
        elif wav.shape[1] < target_len:
            pad_num = target_len - wav.shape[1]
            wav = torch.cat((wav, torch.zeros(1, pad_num)), dim=1)

        return wav

    def generate_augmented_views(self, x):
        """
        Generate two random augmented views of audio sample `x` using attack.py
        """

        def random_augment(audio, sr):
            aug_audio = np.copy(audio)

            if np.random.rand() < 0.5:
                aug_audio = mp3_compression(
                    aug_audio, sr, quality=np.random.choice([2, 4, 6])
                )

            if np.random.rand() < 0.5:
                bit_depth = np.random.choice([8, 16, 24])
                aug_audio = pcm_bit_depth_conversion(aug_audio, sr, pcm=bit_depth)

            if np.random.rand() < 0.5:
                deletion_percentage = np.random.uniform(0.5, 1)
                aug_audio = delete_samples(aug_audio, deletion_percentage)

            if np.random.rand() < 0.5:
                aug_audio = resample(
                    audio=aug_audio,
                    sr=sr,
                    downsample_sr=np.random.choice([16000, 12000, 8000, 4000])
                )

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
        file_key = self.files[idx]["key"]

        # Load audio from S3
        obj = self.s3.get_object(Bucket=self.bucket, Key=file_key)
        wav_bytes = BytesIO(obj["Body"].read())
        wav, sr = torchaudio.load(wav_bytes)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
            sr = self.sample_rate

        if not self.contrastive:
            return {"wav": self.truncate_or_pad(wav), "sample_rate": sr, "name": file_key.split("/")[-1]}
        return {
            "wav": self.truncate_or_pad(wav),
            "augmented_views": self.generate_augmented_views(wav),
            "name": self.files[idx],
        }

    def __len__(self):
        return len(self.files)
