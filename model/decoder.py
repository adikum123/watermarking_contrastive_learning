
import os

import torch
import torch.nn as nn
import torchaudio

from distortions.mel import MelFilterBankLayer
from model.blocks import Conv2Encoder, FCBlock, WatermarkExtracter
from model.utils import istft, stft


class Decoder(nn.Module):

    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set parameters
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.n_fft = process_config["mel"]["n_fft"]
        self.hop_length = process_config["mel"]["hop_length"]
        self.win_length = process_config["mel"]["win_length"]

        # init mel transform and griffin_lim
        self.mel_transform = MelFilterBankLayer(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
        )
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=32,  # Number of iterations for Griffin-Lim
        )

        # init nn modules
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.extractor = WatermarkExtracter(input_channel=1, hidden_dim=model_config["conv2"]["hidden_dim"], block=self.block)
        self.msg_linear_out = FCBlock(win_dim, msg_length)

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        return self.test_forward(x)

    def train_forward(self, x):
        """
        Train forward pass for the decoder. Applies distortion griffin_lim(mel(normalized audio)) and extracts the watermark from both distorted and clean audio.
        Args:
            x (torch.Tensor): Input audio tensor of shape (batch_size, 1, time_steps).
        """
        x_identity = x.clone()

        # === Step 1: Normalize before Mel + GriffinLim ===
        x_norm = x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        # Step 2: Proxy distortion via Mel + GriffinLim
        x_stft, _, _ = stft(
            x_norm.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        x_stft_mag = torch.abs(x_stft)
        x_mel = self.mel_transform(x_stft_mag)
        x_mel_stft_mag_approx = self.inverse_mel(x_mel)
        x_recon = self.griffin_lim(x_mel_stft_mag_approx).unsqueeze(1)

        # Step 4: Extract watermark from distorted audio
        _, spect_dist, _ = stft(
            x_recon.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        feat_dist = self.extractor(spect_dist.abs().to(dtype=torch.float32).unsqueeze(1)).squeeze(1) # abs() to convert to tensor of real values
        msg_feat_dist = torch.mean(feat_dist, dim=2, keepdim=True).transpose(1,2)
        msg_dist = self.msg_linear_out(msg_feat_dist)

        # Step 5: Extract watermark from clean identity
        _, spect_id, _ = stft(
            x_identity.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        feat_id = self.extractor(spect_id.abs().to(dtype=torch.float32).unsqueeze(1)).squeeze(1) # abs() to convert to tensor of real values
        msg_feat_id = torch.mean(feat_id, dim=2, keepdim=True).transpose(1,2)
        msg_id = self.msg_linear_out(msg_feat_id)

        return msg_dist, msg_id

    def get_features(self, x):
        """
        Extract features from the audio input.
        """
        _, spect, _ = stft(
            x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        feat = self.extractor(spect.abs().to(dtype=torch.float32).unsqueeze(1)).squeeze(1)
        return torch.mean(feat, dim=2, keepdim=True).transpose(1,2)

    def test_forward(self, x):
        """
        Test forward pass for the decoder. Applies Mel + GriffinLim and extracts the watermark from the audio.
        """
        _, spect, _ = stft(
            x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        extracted_wm = self.extractor(spect.abs().to(dtype=torch.float32, device=self.device).unsqueeze(1)).squeeze(1)
        msg_features = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        # tesnor (win_dim, batch_size)
        msg = self.msg_linear_out(msg_features)
        return msg

    def save(self, filename="decoder_model.pth"):
        save_dir = "saved_models/regular_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(self, save_path)
        print(f"Model saved to {save_path}")
        print(f"Model saved to {save_path}")
