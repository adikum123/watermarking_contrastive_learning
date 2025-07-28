import torch
from torch import stft as get_stft_output


def stft(x, **kwargs):
    stft_out = get_stft_output(
        x,
        n_fft=kwargs["n_fft"],
        hop_length=kwargs["hop_length"],
        win_length=kwargs["win_length"],
        window=torch.hann_window(kwargs["win_length"], device=x.device),
        return_complex=True,
    )
    # extract magnitude and phase
    spect = torch.abs(stft_out)
    phase = torch.angle(stft_out)
    return stft_out, spect, phase

def istft(spect, phase, num_samples, **kwargs):
    real = spect * torch.cos(phase)
    imag = spect * torch.sin(phase)
    complex_spec = torch.complex(real, imag)
    y = torch.istft(
        complex_spec,
        n_fft=kwargs["n_fft"],
        hop_length=kwargs["hop_length"],
        win_length=kwargs["win_length"],
        window=torch.hann_window(kwargs["win_length"], device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        length=num_samples  # Make sure to match the original time length
    )
    y = y.unsqueeze(1)  # match output shape (B, 1, T)
    return y