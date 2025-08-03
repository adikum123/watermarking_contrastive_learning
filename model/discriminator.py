import torch.nn as nn

from model.blocks import ReluBlock
from model.utils import stft


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            ReluBlock(1,16,3,1,1),
            ReluBlock(16,32,3,1,1),
            ReluBlock(32,64,3,1,1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.linear = nn.Linear(64,1)

        # STFT parameters
        self.n_fft=process_config["mel"]["n_fft"]
        self.hop_length=process_config["mel"]["hop_length"]
        self.win_length=process_config["mel"]["win_length"]

    def forward(self, x):
        _, spect, phase = stft(
            x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        x = spect.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x

    def save(self, filename="discriminator_model.pth"):
        save_dir = "saved_models/regular_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(self, save_path)
        print(f"Model saved to {save_path}")


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param