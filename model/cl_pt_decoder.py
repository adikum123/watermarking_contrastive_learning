import torch
import torch.nn as nn

from model.blocks import FCBlock, ProjectionHead, WatermarkExtracter
from model.frequency import TacotronSTFT, fixed_STFT


class ContrastiveDecoder(nn.Module):
    """
    Class implements a contrastive decoder with projection head
    """
    def __init__(
        self,
        process_config,
        model_config,
        msg_length,
        win_dim,
        mode,
    ):
        super().__init__()
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # mel spectogram and stft
        self.mel_transform = TacotronSTFT(
            filter_length=process_config["mel"]["n_fft"],
            hop_length=process_config["mel"]["hop_length"],
            win_length=process_config["mel"]["win_length"],
        )
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

        # init nn modules
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.extractor = WatermarkExtracter(
            input_channel=1,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=model_config["layer"]["nlayers_decoder"],
        )

        # add projection head
        self.projection_head = ProjectionHead(input_dim=win_dim)
        self.msg_linear_out = FCBlock(self.projection_head.output_dim, msg_length)

        # set mode to contrastive pretrain or finetune
        assert mode in {"contrastive_pretrain", "finetune"}, f"Mode argument unknown: {mode}"
        self.mode = mode

    def forward(self, x):
        """
        Forward pass for decoder
        """
        if self.mode == "contrastive_pretrain":
            return self.get_features(x)
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

        # Step 2: Proxy distortion via Mel + GriffinLim
        x_mel = self.mel_transform.mel_spectrogram(x.squeeze(1))
        x_recon = self.mel_transform.griffin_lim(x_mel).unsqueeze(1)

        # Step 4: Extract watermark from distorted audio
        spect_dist, _ = self.stft.transform(x_recon)
        feat_dist = self.extractor(spect_dist.unsqueeze(1)).squeeze(
            1
        )  # abs() to convert to tensor of real values
        msg_feat_dist = torch.mean(feat_dist, dim=2, keepdim=True).transpose(1, 2)
        msg_dist = self.msg_linear_out(msg_feat_dist)

        # Step 5: Extract watermark from clean identity
        spect_id, _ = self.stft.transform(x_identity)
        feat_id = self.extractor(spect_id.unsqueeze(1)).squeeze(
            1
        )  # abs() to convert to tensor of real values
        msg_feat_id = torch.mean(feat_id, dim=2, keepdim=True).transpose(1, 2)
        msg_id = self.msg_linear_out(msg_feat_id)

        return msg_dist, msg_id

    def get_features(self, x):
        """
        Extract features from the audio input.
        """
        spect, _ = self.stft.transform(x)
        feat = self.extractor(spect.unsqueeze(1)).squeeze(1)
        return self.projection_head(torch.mean(feat, dim=2, keepdim=True).transpose(1, 2))

    def test_forward(self, x):
        """
        Test forward pass for the decoder. Applies Mel + GriffinLim and extracts the watermark from the audio.
        """
        spect, _ = self.stft.transform(x)
        extracted_wm = self.extractor(spect.unsqueeze(1)).squeeze(1)
        msg_features = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1, 2)
        # tesnor (win_dim, batch_size)
        msg = self.msg_linear_out(msg_features)
        return msg
