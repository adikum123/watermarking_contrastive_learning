import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.contrastive_loss import ContrastiveLoss  # adjust import if needed


class LossGradientScaling(nn.Module):
    """
    Computes multiple losses for audio watermarking and scales their gradients
    so each contributes proportionally. Optionally clips gradients.
    """

    def __init__(
        self,
        adversarial,
        contrastive,
        contrastive_loss_type=None,
        clip_grad_norm=None,  # e.g., 1.0 for safe clipping
        beta=1.0,
        eps=1e-6
    ):
        super().__init__()
        self.adversarial = adversarial
        self.contrastive = contrastive
        self.mse = nn.MSELoss()
        if contrastive:
            self.contrastive_loss = ContrastiveLoss(loss_type=contrastive_loss_type)

        # gradient scaling parameters
        self.beta = beta
        self.eps = eps
        self.clip_grad_norm = clip_grad_norm

    def forward(self, embedded, decoded, wav, msg, discriminator_output=None, cl_feat_1=None, cl_feat_2=None):
        """
        Returns a dict of individual losses: embedding, message, adversarial, contrastive
        """
        if self.contrastive:
            assert cl_feat_1 is not None and cl_feat_2 is not None, "Passed None features"
        # Embedding loss
        wm_embedding_loss = self.mse(embedded, wav)

        # Message loss
        if isinstance(decoded, tuple):
            decoded_distorted, decoded_identity = decoded
            message_loss = self.mse(decoded_distorted, msg) + self.mse(decoded_identity, msg)
        else:
            message_loss = self.mse(decoded, msg)

        # Adversarial loss
        adv_loss = torch.tensor(0., device=embedded.device)
        if self.adversarial and discriminator_output is not None:
            labels_real = torch.ones_like(discriminator_output)
            adv_loss = F.binary_cross_entropy_with_logits(discriminator_output, labels_real)

        # Contrastive loss
        contrastive_loss = torch.tensor(0., device=embedded.device)
        if self.contrastive and cl_feat_1 is not None and cl_feat_2 is not None:
            contrastive_loss = self.contrastive_loss(cl_feat_1.squeeze(1), cl_feat_2.squeeze(1))

        return {
            "embedding": wm_embedding_loss,
            "message": message_loss,
            "adversarial": adv_loss,
            "contrastive": contrastive_loss
        }

    def backward(self, losses_dict, params):
        """
        Compute scaled gradients and accumulate in params.
        """
        grads = []
        norms = []

        # 1) Compute per-loss gradients
        for _, loss in losses_dict.items():
            if loss is None or loss == 0:
                grads.append([torch.zeros_like(p) for p in params])
                norms.append(0.0)
                continue

            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            g_list = [p_grad if p_grad is not None else torch.zeros_like(p) for p, p_grad in zip(params, g)]
            grads.append(g_list)

            # Flatten gradient vector to compute norm
            g_vec = torch.cat([p_grad.contiguous().view(-1) for p_grad in g_list])
            norms.append(g_vec.norm(2).item())

        # 2) Compute target norm (mean across all losses)
        n_bar = sum(norms) / len(norms)

        # 3) Compute scaling factors
        scales = [(n_bar / (n + self.eps)) ** self.beta for n in norms]

        # 4) Clear existing gradients
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        # 5) Accumulate scaled gradients
        for scale, g_list in zip(scales, grads):
            for p, p_grad in zip(params, g_list):
                if p.grad is None:
                    p.grad = (scale * p_grad).detach().clone()
                else:
                    p.grad.add_(scale * p_grad.detach())

        # 6) Optional gradient clipping
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
