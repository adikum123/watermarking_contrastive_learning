import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RobustContrastiveLoss(nn.Module):

    def __init__(self, base_model, temperature=0.5, lambda_weight=1/256):
        super().__init__()
        self.base_model = base_model
        self.device = base_model.device
        self.temperature = temperature
        self.lambda_weight = lambda_weight

    def forward(self, pos_1, pos_2):
        pos_1, pos_2 = pos_1.to(self.device), pos_2.to(self.device)
        z1 = self.base_model.forward(pos_1)
        z2 = self.base_model.forward(pos_2)

        with torch.no_grad():
            negatives = torch.cat([z1, z2], dim=0).detach()

        adv_pos_1 = self.generate_adversarial_example(pos_1, pos_2, negatives)
        z1_adv = self.base_model.forward(adv_pos_1)

        loss_rocl = self.contrastive_loss(z1, [z2, z1_adv], negatives)
        loss_reg = self.contrastive_loss(z1_adv, [z2], negatives)
        total_loss = loss_rocl + self.lambda_weight * loss_reg
        return total_loss

    def generate_adversarial_example(self, anchor_img, positive_img, negatives, epsilon=4/255, alpha=1/255, num_iter=10):
        """
        Generates an instance-wise adversarial example from anchor_img (t(x))
        by maximizing the contrastive loss against positive_img (t0(x)) and negatives,
        following the RoCL approach.

        Args:
            anchor_img (Tensor): Anchor view t(x), to be perturbed (batch_size, C, H, W)
            positive_img (Tensor): Positive view t0(x), kept fixed (batch_size, C, H, W)
            negatives (Tensor): Negative embeddings (num_negatives, dim)
            epsilon (float): L-infinity perturbation bound
            alpha (float): PGD step size
            num_iter (int): Number of PGD steps

        Returns:
            adv_img (Tensor): The generated adversarial version of anchor_img
        """
        anchor_img = anchor_img.clone().detach().to(self.device)
        positive_img = positive_img.to(self.device)
        negatives = negatives.to(self.device)

        adv_img = anchor_img.clone().detach().requires_grad_(True)

        # Compute target embedding of the positive image
        with torch.no_grad():
            target_pos = self.base_model.forward(positive_img)  # shape: (batch, dim)

        for _ in range(num_iter):
            adv_embed = self.base_model.forward(adv_img)  # shape: (batch, dim)
            loss = self.contrastive_loss(adv_embed, positives=[target_pos], negatives=negatives)

            self.base_model.zero_grad()
            loss.backward()

            # Gradient step
            grad = adv_img.grad.data
            perturbation = alpha * torch.sign(grad)
            adv_img = adv_img.detach() + perturbation

            # Clamp to ensure within epsilon ball
            eta = torch.clamp(adv_img - anchor_img, min=-epsilon, max=epsilon)
            adv_img = torch.clamp(anchor_img + eta, 0, 1).detach()
            adv_img.requires_grad_()

        return adv_img

    def contrastive_loss(self, anchor, positives, negatives):
        anchor = F.normalize(anchor, dim=1)
        positives = [F.normalize(p, dim=1) for p in positives]
        negatives = F.normalize(negatives, dim=1)

        sim_pos = torch.cat([torch.sum(anchor * p, dim=1, keepdim=True) for p in positives], dim=1)
        sim_neg = anchor @ negatives.T

        sim_pos /= self.temperature
        sim_neg /= self.temperature

        numerator = torch.sum(torch.exp(sim_pos), dim=1)
        denominator = numerator + torch.sum(torch.exp(sim_neg), dim=1)

        loss = -torch.log(numerator / denominator)
        return loss.mean()

class ContrastiveLoss(nn.Module):

    def __init__(self, loss_type="info_nce", temperature=0.5, lambda_param=5e-3):
        """
        Initializes the ContrastiveLoss module.
        Args:
            loss_type: String indicating which loss to use: "info_nce", "nce", "cosine", or "barlow".
            temperature: Temperature parameter for applicable losses.
            lambda_param: Lambda parameter for the Barlow Twins loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.lambda_param = lambda_param

    def forward(self, f1, f2, positive_extra=None):
        if self.loss_type == "info_nce":
            return self.info_nce_loss(f1, f2, positive_extra)
        if self.loss_type == "nce":
            return self.nce_loss(f1, f2)
        if self.loss_type == "cosine":
            return self.cosine_similarity_loss(f1, f2)
        if self.loss_type == "barlow":
            return self.barlow_twins_loss(f1, f2)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def info_nce_loss(self, f1, f2, positive_extra=None):
        """
        Computes the InfoNCE (NT-Xent) loss with optional additional positive examples
        (e.g. adversarial augmentations) as in RoCL.
        """
        batch_size = f1.size(0)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        if positive_extra is not None:
            positive_extra = F.normalize(positive_extra, dim=1)
            anchors = torch.cat([f1, f1], dim=0)         # (2N, d)
            positives = torch.cat([f2, positive_extra], dim=0)  # (2N, d)
        else:
            anchors = f1
            positives = f2

        # Total features for similarity calculation
        features = torch.cat([anchors, positives], dim=0)  # (4N or 2N, d)
        sim_matrix = torch.matmul(anchors, features.T) / self.temperature  # (2N, 4N) or (N, 2N)

        # Construct positive indices: anchor[i] matches positive[i]
        labels = torch.arange(anchors.size(0), device=f1.device)
        positive_indices = labels + anchors.size(0)  # offset by number of anchors

        # Mask self-similarity in the anchor block (anchors compared to anchors)
        mask = torch.zeros_like(sim_matrix, dtype=torch.bool)  # (2N, 4N)
        num_anchors = anchors.size(0)
        mask[:, :num_anchors] = torch.eye(num_anchors, device=f1.device).bool()

        # Apply mask
        sim_matrix.masked_fill_(mask, -1e9)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, positive_indices)
        return loss

    def nce_loss(self, f1, f2):
        """
        Computes the Noise Contrastive Estimation (NCE) loss.
        """
        batch_size = f1.size(0)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        logits = torch.matmul(f1, f2.T) / self.temperature  # (N, N) similarity matrix
        positives = torch.diag(logits)
        logsumexp = torch.logsumexp(logits, dim=1)
        loss = - (positives - logsumexp).mean()
        return loss

    def cosine_similarity_loss(self, f1, f2):
        """
        Computes a loss based on the negative cosine similarity.
        """
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        loss = - (f1 * f2).sum(dim=1).mean()
        return loss

    def off_diagonal(self, x):
        """
        Returns a flattened view of the off-diagonal elements of a square matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_twins_loss(self, f1, f2):
        """
        Computes the Barlow Twins loss.
        """
        batch_size = f1.size(0)
        f1_norm = (f1 - f1.mean(0)) / f1.std(0)
        f2_norm = (f2 - f2.mean(0)) / f2.std(0)
        c = torch.mm(f1_norm.T, f2_norm) / batch_size  # Cross-correlation matrix
        # On-diagonal: encourage values to be 1
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        # Off-diagonal: encourage values to be 0
        off_diag = self.off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss
