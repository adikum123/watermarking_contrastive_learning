import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.contrastive_loss import ContrastiveLoss


class WatermarkLoss(nn.Module):

    def __init__(
        self,
        lambda_e,
        lambda_m,
        lambda_a,
        lambda_cl,
        adversarial,
        contrastive,
        contrastive_loss_type,
        decay_rate_m=0.5,  # faster decay for message loss
        decay_rate_cl=0.5,  # slower decay for contrastive loss
    ):
        """
        Custom loss for audio watermarking.

        Args:
            lambda_e (float): weight for embedding loss
            lambda_m (float): weight for message loss
            lambda_a (float): weight for adversarial loss
            adversarial (bool): whether to include adversarial loss
        """
        super(WatermarkLoss, self).__init__()
        self.lambda_e = lambda_e
        self.lambda_m = lambda_m
        self.lambda_a = lambda_a if adversarial else 0.0
        self.lambda_cl = lambda_cl if contrastive else 0.0
        self.lambda_m_initial = lambda_m
        self.lambda_cl_initial = lambda_cl if contrastive else 0.0
        self.adversarial = adversarial
        self.mse = nn.MSELoss()
        self.contrastive = contrastive
        if contrastive:
            # init contrastive loss
            self.contrastive_loss = ContrastiveLoss(loss_type=contrastive_loss_type)

        # decay rates
        self.decay_rate_m = decay_rate_m
        self.decay_rate_cl = decay_rate_cl

    def forward(
        self,
        embedded,
        decoded,
        wav,
        msg,
        discriminator_output=None,
        cl_feat_1=None,
        cl_feat_2=None,
    ):
        """
        Compute the total loss.

        Args:
            embedded (Tensor): watermarked audio
            decoded (tuple or Tensor): (distorted_decoded, identity_decoded) or just decoded message
            wav (Tensor): original audio
            msg (Tensor): ground truth watermark
            discriminator_output (Tensor, optional): output from discriminator (for adv loss)

        Returns:
            Tensor: total weighted loss
        """
        # Embedding loss
        wm_embedding_loss = self.mse(embedded, wav)

        # Message loss
        if isinstance(decoded, tuple):
            decoded_distorted, decoded_identity = decoded
            message_loss = self.mse(decoded_distorted, msg) + self.mse(
                decoded_identity, msg
            )
        else:
            message_loss = self.mse(decoded, msg)

        # Adversarial loss
        adv_loss = 0
        if self.adversarial and discriminator_output is not None:
            labels_real = torch.ones_like(discriminator_output)
            adv_loss = F.binary_cross_entropy_with_logits(
                discriminator_output, labels_real
            )

        # contrastive loss
        contrastive_loss = 0
        if self.contrastive:
            contrastive_loss = self.contrastive_loss(
                cl_feat_1.squeeze(1), cl_feat_2.squeeze(1)
            )

        # Total loss
        total_loss = (
            self.lambda_e * wm_embedding_loss
            + self.lambda_m * message_loss
            + self.lambda_a * adv_loss
            + self.lambda_cl * contrastive_loss
        )
        return total_loss

    def discriminator_loss(self, curr_bs, device, discriminator, embedded, wav):
        labels_real = torch.full((curr_bs, 1), 1, device=device).float()
        labels_fake = torch.full((curr_bs, 1), 0, device=device).float()
        discriminator_output_wav = discriminator(wav)
        discriminator_output_embedded = discriminator(embedded.detach())

        # get adversarial loss on real audio
        discriminator_adv_loss_wav = F.binary_cross_entropy_with_logits(
            discriminator_output_wav, labels_real
        )
        discriminator_adv_loss_wav.backward()

        # get adversarial loss on embedded and perform step
        discriminator_adv_loss_embedded = F.binary_cross_entropy_with_logits(
            discriminator_output_embedded, labels_fake
        )
        discriminator_adv_loss_embedded.backward()

    def schedule_lambdas(self, epoch):
        # Exponential decay per epoch
        self.lambda_m = self.lambda_m_initial * (self.decay_rate_m**epoch)
        print(f"Lambda m: {self.lambda_m:.6f}")
        if self.contrastive:
            self.lambda_cl = self.lambda_cl_initial * (self.decay_rate_cl**epoch)
            print(f"Lambda cl: {self.lambda_cl:.6f}")
