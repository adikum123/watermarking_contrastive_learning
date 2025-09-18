import os

import torch
import torchaudio
from torch.utils.data import Dataset


class LjAudioDataset(Dataset):
    """
    Loads lj speech audio dataset
    """

    def __init__(self, process_config, split="train"):
        self.ljspeech_root = os.path.join(
            "mnt", "s3", "data", "lj_speech_dataset", "LJSpeech-1.1/"
        )
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.set_data()

    def set_data(self):
        # Find metadata file
        metadata_path = os.path.join(self.ljspeech_root, "metadata.csv")
        with open(metadata_path, encoding="utf-8") as f:
            lines = [line.strip().split("|") for line in f]
        # Each entry: (wav_file, normalized_text, original_text)
        files = [
            {"path": os.path.join(self.ljspeech_root, "wavs", line[0] + ".wav")}
            for line in lines
        ]

        # Create simple split: ~12k train, 500 val, 500 test
        if self.split == "train":
            self.files = files[:12000]
        elif self.split == "val":
            self.files = files[12000:12500]
        else:  # test
            self.files = files[12500:]

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
