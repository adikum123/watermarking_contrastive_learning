import random
import shutil
import subprocess
import sys

from distortions.attacks.attacks.cut_samples.attack import CutSamplesAttack
from distortions.attacks.attacks.equalizer.attack import EqualizerAttack
from distortions.attacks.attacks.gaussian_noise.attack import \
    GaussianNoiseAttack
from distortions.attacks.attacks.resampling_poly.attack import \
    ResamplingPolyAttack
from distortions.attacks.attacks.time_stretch.attack import TimeStretchAttack


class ContrastiveAugmentations:

    def __init__(self, sr):
        self.sr = sr
        ContrastiveAugmentations.check_rubberband()
        self.attack_list = ContrastiveAugmentations.init_attack_list()

    @staticmethod
    def check_rubberband():
        """Check if rubberband-cli is installed; install if missing."""
        if shutil.which("rubberband") is not None:
            print("rubberband-cli is already installed.")
            return True

        print("rubberband-cli not found. Attempting to install...")
        try:
            # Linux / Debian-based
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "rubberband-cli"], check=True)
            print("rubberband-cli installed successfully.")
        except Exception as e:
            print(f"Failed to install rubberband-cli automatically: {e}")
            print("Please install it manually: https://breakfastquay.com/rubberband/")
            return False

        return shutil.which("rubberband") is not None

    @staticmethod
    def init_attack_list():
        return [
            {
                "prob": 1,
                "attack": CutSamplesAttack()
            },
            {
                "prob": 0.6,
                "attack": ResamplingPolyAttack()
            },
            {
                "prob": 0.4,
                "attack": TimeStretchAttack()
            },
            {
                "prob": 0.6,
                "attack": EqualizerAttack()
            },
            {
                "prob": 0.7,
                "attack": GaussianNoiseAttack()
            }
        ]

    def apply(self, audio):
        for attack in self.attack_list:
            if random.random() <= attack["prob"]:
                audio = attack["attack"].apply(audio=audio, sampling_rate=self.sr)
        return audio
