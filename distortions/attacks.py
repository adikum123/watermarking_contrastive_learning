import os
import platform
import subprocess
import tempfile
import time

import librosa
import numpy as np
import pyrubberband as pyrb
import pywt
import soundfile as sf
from scipy.signal import lfilter


def pcm_bit_depth_conversion(audio, sr, pcm=16):
    """
    Simulate MP3 compression with PCM bit depth conversion

    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
        pcm: PCM bit depth (8, 16, 24)
        quality: MP3 quality (0=best, 9=worst)
    """
    # Convert to specified PCM bit depth and back (simulates quantization)
    if pcm == 8:
        # 8-bit signed: -128 to 127
        audio_int = np.clip(audio * 127.0, -128, 127).astype(np.int8)
        audio = audio_int.astype(np.float32) / 127.0
    elif pcm == 16:
        # 16-bit signed: -32768 to 32767
        audio_int = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        audio = audio_int.astype(np.float32) / 32767.0
    elif pcm == 24:
        # 24-bit signed: -8388608 to 8388607
        audio_int = np.clip(audio * 8388607.0, -8388608, 8388607).astype(np.int32)
        audio = audio_int.astype(np.float32) / 8388607.0
    else:
        raise ValueError(f"Unsupported PCM bit depth: {pcm}")
    return audio


def mp3_compression(audio, sr, quality=2):
    """
    MP3 compression

    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
        quality: MP3 quality (0=best, 9=worst) - 2 is good quality
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg not found. Please install FFmpeg or skip MP3 tests.")

    def safe_delete(filepath, max_retries=5):
        """Safely delete a file with retries for Windows file locking issues"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    print(
                        f"Warning: Could not delete {filepath} after {max_retries} attempts"
                    )

    # Create temporary files
    temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
    temp_mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")

    try:
        # Close file descriptors immediately to avoid conflicts
        os.close(temp_wav_fd)
        os.close(temp_mp3_fd)

        # Apply PCM conversion and save
        audio = pcm_bit_depth_conversion(audio, sr, 16)
        sf.write(temp_wav_path, audio, sr)

        # Convert to MP3 using ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-i", temp_wav_path, "-q:a", str(quality), mp3_path, "-y"],
            capture_output=True,
            check=True,
        )

        # Small delay to ensure FFmpeg fully releases files
        time.sleep(0.1)

        # Load the MP3 file
        audio_data, sample_rate = sf.read(mp3_path)

        # Another small delay before cleanup
        time.sleep(0.1)

        return audio_data

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
        )
    except Exception as e:
        raise e
    finally:
        # Clean up temporary files with retry logic
        safe_delete(temp_wav_path)
        safe_delete(mp3_path)


def delete_samples(audio, percentage):
    """
    Delete a percentage of samples from the audio.

    Args:
        audio: Input audio (1D np.ndarray, float32, range -1 to 1)
        percentage: Fraction of samples to delete (0.0–1.0)
    """
    length = len(audio)
    samples_to_delete = int(percentage * length)

    # Edge case: delete everything
    if samples_to_delete >= length:
        return np.array([], dtype=audio.dtype)

    start_delete = np.random.randint(0, length - samples_to_delete)
    end_delete = start_delete + samples_to_delete

    return np.concatenate([audio[:start_delete], audio[end_delete:]])


def resample(audio, sr, downsample_sr=8000, print_stmt=False):
    """
    Resample audio to 16kHz and back to original rate using linear interpolation

    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
    """
    # Downsample to 16kHz
    downsample_factor = sr // downsample_sr
    if downsample_factor > 1:
        # Simple decimation (take every nth sample)
        downsampled_audio = audio[::downsample_factor]
        if print_stmt:
            print(
                f"Downsampled to {downsample_sr} Hz: {len(downsampled_audio)} samples"
            )

        # Upsample back to original rate using linear interpolation
        upsampled_audio = np.interp(
            np.arange(len(audio)),
            np.arange(0, len(audio), downsample_factor),
            downsampled_audio,
        )
        if print_stmt:
            print(f"Upsampled back to {sr} Hz: {len(upsampled_audio)} samples")
        return upsampled_audio

    if print_stmt:
        print(f"Audio already at or below 16kHz ({sr} Hz), skipping resampling")
    return audio


def wavelet(audio, wavelet="db1", wt_mode="soft", threshold_factor=1.0):
    """
    Perform wavelet-based denoising on an audio signal.

    Args:
        audio (np.ndarray): Input audio signal.
        wavelet (str): Wavelet type (e.g., 'db1', 'sym5'). Default is 'db1'.
        wt_mode (str): Thresholding mode ('soft' or 'hard'). Default is 'soft'.
        threshold_factor (float): Scaling factor for the universal threshold used in wavelet denoising.
            - Controls denoising aggressiveness:
                - < 1.0 → less aggressive, preserves more details (e.g., 0.5, 0.8)
                - 1.0 → standard universal threshold (default)
                - > 1.0 → more aggressive, removes more noise but may distort audio (e.g., 1.5, 2.0, 3.0)

    Returns:
        np.ndarray: The denoised audio signal.

    Examples:
        denoised_audio = wavelet(audio, wavelet="db4", wt_mode="soft", threshold_factor=0.8)
        denoised_audio = wavelet(audio, wavelet="sym5", wt_mode="hard", threshold_factor=2.0)
    """
    threshold = compute_threshold(audio, wavelet, threshold_factor)

    coeffs = pywt.wavedec(audio, wavelet)
    coeffs_denoised = [pywt.threshold(c, threshold, mode=wt_mode) for c in coeffs]

    denoised_audio = pywt.waverec(coeffs_denoised, wavelet)

    return denoised_audio

def compute_threshold(audio, wavelet, threshold_factor):
    """
    Compute the universal threshold for wavelet-based denoising.

    Args:
        audio (np.ndarray): Input audio signal.
        wavelet (str): Wavelet type (e.g., 'db1', 'sym5', etc.) used for decomposition.
        threshold_factor (float): Threshold factor for the universal threshold.

    Returns:
        float: The calculated threshold value.

    Notes:
        - This function uses the universal threshold formula:
            Threshold = sigma * sqrt(2 * log(n)),
            where sigma is the noise standard deviation estimated from the detail coefficients,
            and n is the length of the audio signal.
        - The estimation of sigma uses the robust formula:
            sigma = median(|coeffs[-1]|) / 0.6745,
            which is based on the assumption of Gaussian white noise.
        - The universal threshold is particularly effective for denoising signals corrupted by
            additive white Gaussian noise.

    """
    coeffs = pywt.wavedec(audio, wavelet)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(audio))) * threshold_factor
    return threshold

def pitch_shift(audio, sr=None, cents=None):
    """
    Perform pitch shifting using pyrubberband.
    Falls back to librosa if rubberband-cli is not installed.

    Args:
        audio (np.ndarray): Input audio signal.
        **kwargs: Additional parameters for pitch shifting.
            - sampling_rate (int): Sampling rate of the audio in Hz (optional).
            - cents (float): Pitch shift in cents (1 cent = 1/100 of a semitone) (optional).

    Returns:
        np.ndarray: The pitch-shifted audio signal.

    Notes:
        - This function uses the pyrubberband library, which provides high-quality
        pitch shifting without altering the speed of the audio.
        - If rubberband-cli is not installed, falls back to librosa.effects.pitch_shift
        - Ensure that pyrubberband and the Rubber Band Library are installed before use.
    """
    if sr is None or cents is None:
        raise ValueError(
            "A sampling_rate and cents must be specified for PitchShiftAttack."
        )

    semitones = cents / 100

    try:
        # Try using pyrubberband first
        return pyrb.pitch_shift(audio, sr, semitones)
    except Exception as e:
        print(f"Pyrubberband failed: {str(e)}. Falling back to librosa pitch_shift.")
        # Use librosa as a fallback
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def gaussian_noise(audio, snr_db):
    """
    Perform a Gaussian noise attack on an audio signal. Adds random noise constrained by SNR.
    Args:
        audio (np.ndarray): The input audio signal.
        **kwargs: Additional parameters for the Gaussian noise attack:
            - snr_db (float): Desired Signal-to-Noise Ratio in dB
    Returns:
        np.ndarray: The processed audio signal with the Gaussian noise applied.

    """
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
    audio_noisy = audio + noise


    #actual_noise_power = np.mean((audio_noisy - audio) ** 2)
    #actual_snr = 10 * np.log10(signal_power / actual_noise_power)
    #print(f"Target SNR: {snr_db} dB, Achieved SNR: {actual_snr:.2f} dB")

    return audio_noisy

def pink_noise(audio, amplitude):
    """
    Perform a pink noise (Voss-McCartney) attack on an audio signal.
    Args:
        audio (np.ndarray): The input audio signal.
        **kwargs: Additional parameters for the pink noise attack:
            - amplitude (float): Controls how loud the sound is.
    Returns:
        np.ndarray: The processed audio signal with the pink noise applied.

    """
    n_samples = len(audio)
    white = np.random.normal(0, 1, n_samples)

    # Apply pink noise filter (from Julius O. Smith / Audio EQ Cookbook)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)

    pink = pink / np.max(np.abs(pink)) * amplitude
    pink = pink.astype(np.float32)

    noisy_audio=audio+pink

    return noisy_audio