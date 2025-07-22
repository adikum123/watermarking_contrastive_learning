import os

import librosa
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class AudioDatasetTF:
    def __init__(self, process_config, dataset_path, split="train", batch_size=100):
        self.dataset_path = "cv-corpus-22.0-delta-2025-06-20-en/cv-corpus-22.0-delta-2025-06-20/en/clips"
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]  # patch size
        self.max_len = process_config["audio"]["max_len"]  # in samples
        self.batch_size = batch_size

        self.fixed_num_patches = self.max_len // self.win_len # fiksiraj broj patch-eva
        self.fixed_len = self.fixed_num_patches * self.win_len # fiksiraj duzinu

        # List audio files and split
        all_files = sorted(os.listdir(self.dataset_path))
        split_idx = int(len(all_files) * 0.8)
        assert split in ("train", "val"), "Invalid split"
        self.file_list = all_files[:split_idx] if split == "train" else all_files[split_idx:]

        # Create tf.data.Dataset
        self.dataset = tf.data.Dataset.from_generator(
            lambda: self.audio_generator(),
            output_signature={
                "matrix": tf.TensorSpec(shape=(self.fixed_num_patches, self.win_len), dtype=tf.float32), # matrica iste duzine jer je fiksirana
                "sample_rate": tf.TensorSpec(shape=(), dtype=tf.int32),
                "patch_num": tf.TensorSpec(shape=(), dtype=tf.int32),
                "pad_num": tf.TensorSpec(shape=(), dtype=tf.int32),
                "name": tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )

        # Batch (no padded_batch needed, shape is already fixed)
        self.dataset = self.dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def audio_generator(self):
        for audio_name in self.file_list:
            file_path = os.path.join(self.dataset_path, audio_name)
            wav, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            print(f"Wave: {wav.shape}, Sample Rate: {sr}, Name: {audio_name}")
            # Trim or pad waveform to fixed_len
            wav = wav[:self.fixed_len]
            pad_num = self.fixed_len - len(wav)
            if pad_num > 0:
                wav = np.pad(wav, (0, pad_num), mode="constant")

            patch_num = len(wav) // self.win_len
            wav_matrix = wav.reshape(-1, self.win_len)  # shape: [num_patches, win_len]

            yield {
                "matrix": wav_matrix.astype(np.float32),
                "sample_rate": np.int32(sr),
                "patch_num": np.int32(patch_num),
                "pad_num": np.int32(pad_num),
                "name": audio_name.encode()
            }


if __name__ == "__main__":
    # Example configs
    process_config = {
        "audio": {
            "sample_rate": 16000,
            "max_wav_value": 32768.0,
            "win_len": 1024,
            "max_len": 16000 * 10  # e.g. 10 seconds max
        }
    }

    train_config = {
        "dataset": "cv-corpus",
        "path": {
            "raw_path": "cv-corpus-22.0-delta-2025-06-20-en/cv-corpus-22.0-delta-2025-06-20/en/clips"
        }
    }

    # Create dataset
    train_ds = AudioDatasetTF(process_config, train_config, split="train")

    # Iterate and print some samples
    for batch in train_ds.dataset.take(1):
        print("Batch matrix shape:", batch["matrix"].shape)
        print("Sample rate:", batch["sample_rate"])
        print("Patch num:", batch["patch_num"])
        print("Pad num:", batch["pad_num"])
        print("Names:", [name.decode() for name in batch["name"].numpy()])
        print("Names:", [name.decode() for name in batch["name"].numpy()])
