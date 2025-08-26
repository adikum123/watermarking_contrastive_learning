import os
import json
import pandas as pd
from mutagen.mp3 import MP3
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


sample_rate_counts = defaultdict(int)
durations = []  # store audio lengths in seconds


data_dir = os.path.join("mnt", "s3", "data", "raw")
clips_dir = os.path.join(data_dir, "clips")


def count_sr(tsv_file, clips_dir, frac=0.2, random_state=42):
    df = pd.read_csv(tsv_file, sep="\t")
    # sample 20% of rows (frac=0.2), reproducible with random_state
    df = df.sample(frac=frac, random_state=random_state)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(tsv_file)}"):
        audio_file = row["path"]
        input_path = os.path.join(clips_dir, audio_file)
        audio = MP3(input_path)
        sr = audio.info.sample_rate
        sample_rate_counts[sr] += 1
        duration = audio.info.length
        durations.append(duration)


count_sr(os.path.join(data_dir, "dev.tsv"), clips_dir)
count_sr(os.path.join(data_dir, "test.tsv"), clips_dir)
#count_sr(os.path.join(data_dir, "train.tsv"), clips_dir, frac=0.01)
print(sample_rate_counts)
for sample_rate, count in sample_rate_counts.items():
    print(f"Sample Rate: {sample_rate} Hz - Count: {count}")
output_file = "dev_sample_rate_counts.json"
with open(output_file, "w") as f:
    json.dump(sample_rate_counts, f, indent=4)

# ---- Plot distribution of audio lengths ----
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=100, edgecolor="black")
plt.title("Distribution of Audio Lengths")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of clips")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save plot instead of showing
plot_file = "audio_length_distribution.png"
plt.savefig(plot_file)
print(f"Saved plot to {plot_file}")