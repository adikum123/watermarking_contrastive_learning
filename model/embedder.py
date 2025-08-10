import os

import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from model.blocks import Conv2Encoder, FCBlock, WatermarkEmbedder
from model.utils import istft, stft


class Embedder(nn.Module):

    def __init__(
        self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1
    ):
        super(Embedder, self).__init__()
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set parameters
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]

        # STFT parameters
        self.n_fft=process_config["mel"]["n_fft"]
        self.hop_length=process_config["mel"]["hop_length"]
        self.win_length=process_config["mel"]["win_length"]

        # Encoder, embedder and message linear layer
        self.msg_linear_in = FCBlock(msg_length, win_dim, activation=LeakyReLU(inplace=True))
        self.encoder = Conv2Encoder(input_channel=1, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_CE)
        self.embedder = WatermarkEmbedder(input_channel=self.EM_input_dim, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_EM)

    def forward(self, x, msg):
        _, spect, phase = stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

        # encode carrier spectrogram
        carrier_encoded = self.encoder(spect.unsqueeze(1))

        # encode watermark message
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])

        # concatenate features and embed concatenated features
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)
        carrier_wateramrked = self.embedder(concatenated_feature)

        # inverse STFT to get the watermarked audio
        y = istft(
            carrier_wateramrked.squeeze(1),
            phase.squeeze(1),
            x.shape[2],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        return y, carrier_wateramrked

    def save(self, save_dir="saved_models/regular_models", filename="embedder_model.pth"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(self, save_path)
        print(f"Model saved to {save_path}")
