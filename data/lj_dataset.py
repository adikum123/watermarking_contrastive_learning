from io import BytesIO
from typing import Tuple

import boto3
import torch
import torchaudio
from torch.utils.data import Dataset

from distortions.attack_performer import AttackPerformer


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
        self.attack_performer = AttackPerformer() if self.contrastive else None

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

        # Remove channel dim (1, N) → (N,)
        x_np = x.squeeze().numpy()

        # Apply exactly one augmentation per view
        view1 = self.attack_performer.get_contrastive_views(x=x_np, sr=self.sample_rate)
        view2 = self.attack_performer.get_contrastive_views(x=x_np, sr=self.sample_rate)

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
