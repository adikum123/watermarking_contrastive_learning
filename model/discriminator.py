import torch.nn as nn

from model.blocks import ReluBlock
from model.frequency import fixed_STFT

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

        # STFT
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x):
        spect, phase = self.stft.transform(x)
        x = spect.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param