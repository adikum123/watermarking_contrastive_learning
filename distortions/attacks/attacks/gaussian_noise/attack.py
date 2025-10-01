import numpy as np

from distortions.attacks.attacks.base_attack import BaseAttack


class GaussianNoiseAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a Gaussian noise attack on an audio signal. Adds random noise constrained by SNR.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the Gaussian noise attack:
                - snr_db (float): Desired Signal-to-Noise Ratio in dB
        Returns:
            np.ndarray: The processed audio signal with the Gaussian noise applied.

        """

        snr_db = kwargs.get(
            "snr_db", self.config.get("snr_db")
        )
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
        audio_noisy = audio + noise

        return audio_noisy