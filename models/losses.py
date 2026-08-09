# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
models/losses.py
----------------
Custom loss classes used by both EDSR and LaMa training:
  PerceptualLoss    – VGG19 feature-level MSE loss (frozen backbone)
  ReconstructionLoss – weighted L1 + perceptual combination
  AdversarialLoss   – hinge GAN loss (for LaMa discriminator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# Perceptual Loss (VGG19)
# ---------------------------------------------------------------------------


class PerceptualLoss(nn.Module):
    """
    Computes the perceptual (feature matching) loss between *pred* and *target*
    using a pre-trained, frozen VGG19.

    Feature layers tapped:
      relu1_2  (block1_conv2, idx=3  in features)
      relu2_2  (block2_conv2, idx=8)
      relu3_3  (block3_conv3, idx=15)

    The loss is the sum of MSE between activations at each layer,
    weighted equally by default.

    Parameters
    ----------
    layer_weights : dict mapping layer index → float weight.
                    Defaults to {3: 1.0, 8: 1.0, 15: 1.0}.
    device        : where the VGG19 backbone lives.  If None, auto-detected.
    """

    # Mapping friendly name → sequential index in vgg19.features
    _DEFAULT_LAYERS = {3: 1.0, 8: 1.0, 15: 1.0}

    # ImageNet normalisation expected by VGG
    _MEAN = torch.tensor([0.485, 0.456, 0.406])
    _STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, layer_weights: dict = None, device: torch.device = None):
        super().__init__()
        self.layer_weights = layer_weights or self._DEFAULT_LAYERS
        max_idx = max(self.layer_weights.keys()) + 1

        # Load VGG19 pre-trained on ImageNet; freeze all parameters
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg.features.children())[:max_idx]
        self.vgg_slice = nn.Sequential(*features)
        for p in self.vgg_slice.parameters():
            p.requires_grad = False
        self.vgg_slice.eval()

        # Register mean/std as buffers so they move with .to(device)
        self.register_buffer("mean", self._MEAN.view(1, 3, 1, 1))
        self.register_buffer("std", self._STD.view(1, 3, 1, 1))

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise images from [0, 1] to ImageNet statistics."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
        pred, target : (B, 3, H, W) float tensors in [0, 1]

        Returns:
        Scalar loss tensor.
        """
        pred_n = self._normalise(pred.clamp(0, 1))
        target_n = self._normalise(target.clamp(0, 1))

        loss = torch.tensor(0.0, device=pred.device)
        p, t = pred_n, target_n

        for idx, layer in enumerate(self.vgg_slice):
            p = layer(p)
            t = layer(t)
            if idx in self.layer_weights:
                w = self.layer_weights[idx]
                loss = loss + w * F.mse_loss(p, t.detach())

        return loss


# ---------------------------------------------------------------------------
# Style Consistency Loss
# ---------------------------------------------------------------------------


class StyleLoss(nn.Module):
    """
    Computes style consistency loss (difference between Gram matrices of VGG19 features).
    Ensures the restored regions match the paint brushstrokes, texture, and colour
    palette of the undamaged parts.
    """

    def __init__(self, layer_weights: dict = None, device: torch.device = None):
        super().__init__()
        # Share standard PerceptualLoss initialization to get VGG19
        self.perceptual = PerceptualLoss(layer_weights=layer_weights, device=device)

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
        pred, target : (B, 3, H, W) float tensors in [0, 1]

        Returns:
        Scalar loss tensor.
        """
        pred_n = self.perceptual._normalise(pred.clamp(0, 1))
        target_n = self.perceptual._normalise(target.clamp(0, 1))

        loss = torch.tensor(0.0, device=pred.device)
        p, t = pred_n, target_n

        for idx, layer in enumerate(self.perceptual.vgg_slice):
            p = layer(p)
            t = layer(t)
            if idx in self.perceptual.layer_weights:
                w = self.perceptual.layer_weights[idx]
                g_p = self._gram_matrix(p)
                g_t = self._gram_matrix(t)
                loss = loss + w * F.mse_loss(g_p, g_t.detach())

        return loss


# ---------------------------------------------------------------------------
# Reconstruction Loss
# ---------------------------------------------------------------------------


class ReconstructionLoss(nn.Module):
    """
    Weighted combination of L1 pixel loss, perceptual VGG19 loss, and style loss.

    loss = l1_weight * L1(pred, target)
         + perceptual_weight * PerceptualLoss(pred, target)
         + style_weight * StyleLoss(pred, target)

    Parameters
    ----------
    l1_weight          : weight for the L1 term.          Default 1.0
    perceptual_weight  : weight for the perceptual term.  Default 0.1
    style_weight       : weight for the style loss term.       Default 0.0 (disabled)
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        style_weight: float = 0.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.perceptual = PerceptualLoss()
        if style_weight > 0:
            self.style = StyleLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
        total_loss : scalar tensor
        breakdown  : dict {'l1': float, 'perceptual': float, 'style': float} for logging
        """
        l1 = F.l1_loss(pred, target)
        perc = self.perceptual(pred, target)
        total = self.l1_weight * l1 + self.perceptual_weight * perc
        breakdown = {"l1": l1.item(), "perceptual": perc.item()}

        if self.style_weight > 0:
            s_loss = self.style(pred, target)
            total = total + self.style_weight * s_loss
            breakdown["style"] = s_loss.item()

        return total, breakdown


# ---------------------------------------------------------------------------
# Adversarial (Hinge) Loss
# ---------------------------------------------------------------------------


class AdversarialLoss(nn.Module):
    """
    Hinge GAN loss used by LaMa.

    For the **discriminator**:
        L_D = max(0, 1 - D(real)) + max(0, 1 + D(fake))

    For the **generator**:
        L_G = -mean(D(fake))

    Usage
    -----
    loss_fn = AdversarialLoss()

    # ---- discriminator step ----
    d_loss = loss_fn(d_real_logits, d_fake_logits, mode="discriminator")

    # ---- generator step ----
    g_loss = loss_fn(d_fake_logits, mode="generator")
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_logits: torch.Tensor = None,
        fake_logits: torch.Tensor = None,
        mode: str = "discriminator",
    ) -> torch.Tensor:
        """
        Args:
        real_logits : discriminator output on real images (needed for 'discriminator' mode)
        fake_logits : discriminator output on generated images
        mode        : 'discriminator' or 'generator'

        Returns:
        Scalar loss tensor.
        """
        if mode == "discriminator":
            if real_logits is None or fake_logits is None:
                raise ValueError(
                    "Both real_logits and fake_logits needed for discriminator loss."
                )
            # Hinge: penalise if D(real) < 1  or  D(fake) > -1
            loss_real = F.relu(1.0 - real_logits).mean()
            loss_fake = F.relu(1.0 + fake_logits).mean()
            return loss_real + loss_fake

        elif mode == "generator":
            if fake_logits is None:
                raise ValueError("fake_logits needed for generator loss.")
            # Generator wants D(fake) to be as large as possible
            return -fake_logits.mean()

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'discriminator' or 'generator'."
            )
