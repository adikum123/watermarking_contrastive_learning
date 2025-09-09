import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from model.blocks import Conv2Encoder, FCBlock, WatermarkEmbedder
from model.frequency import fixed_STFT


class Embedder(nn.Module):

    def __init__(
        self,
        process_config,
        model_config,
        msg_length,
        win_dim,
    ):
        super(Embedder, self).__init__()
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set parameters
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 2

        # stft transform
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

        # Encoder, embedder and message linear layer
        self.msg_linear_in = FCBlock(
            msg_length, win_dim, activation=LeakyReLU(inplace=True)
        )
        self.encoder = Conv2Encoder(
            input_channel=1,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=model_config["layer"]["nlayers_encoder"],
        )
        self.embedder = WatermarkEmbedder(
            input_channel=self.EM_input_dim,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=model_config["conv2"]["layers_EM"],
        )

    def forward(self, x, msg):
        num_samples = x.shape[2]

        # encode carrier spectrogram
        spect, phase = self.stft.transform(x)
        carrier_encoded = self.encoder(spect.unsqueeze(1))

        # encode watermark message
        watermark_encoded = (
            self.msg_linear_in(msg)
            .transpose(1, 2)
            .unsqueeze(1)
            .repeat(1, 1, 1, carrier_encoded.shape[3])
        )

        # concatenate features and embed concatenated features
        concatenated_feature = torch.cat(
            (carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1
        )
        carrier_wateramrked = self.embedder(concatenated_feature)

        # inverse STFT to get the watermarked audio
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked
