# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
utils/metrics.py
----------------
Image quality metrics for evaluating restoration quality:
  - PSNR  : Peak Signal-to-Noise Ratio       (higher is better; good ≥ 30 dB)
  - SSIM  : Structural Similarity Index      (higher is better; good ≥ 0.85)
  - LPIPS : Learned Perceptual Image Patch   (lower is better; good ≤ 0.20)
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# lpips is installed via: pip install lpips
try:
    import lpips as lpips_lib

    _LPIPS_NET = None  # lazy-loaded on first use
except ImportError:
    lpips_lib = None
    _LPIPS_NET = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy_uint8(img) -> np.ndarray:
    """
    Convert a tensor or numpy array to uint8 HxWxC numpy array in [0, 255].
    Accepts:
      - torch.Tensor: shape (C, H, W) or (1, C, H, W) in [0, 1]
      - np.ndarray:   shape (H, W, C) in [0, 1] or [0, 255]
    """
    if isinstance(img, Tensor):
        img = img.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)  # remove batch dim
        img = img.permute(1, 2, 0).numpy()  # C,H,W → H,W,C

    img = np.array(img, dtype=np.float32)

    if img.max() <= 1.0 + 1e-6:  # [0, 1] range
        img = (img * 255.0).clip(0, 255)

    return img.astype(np.uint8)


def _to_lpips_tensor(img, device: torch.device) -> Tensor:
    """
    Convert uint8 HxWxC numpy or tensor to LPIPS-expected
    float tensor in [-1, 1] with shape (1, C, H, W).
    """
    if isinstance(img, Tensor):
        t = img.float()
        if t.ndim == 3:
            t = t.unsqueeze(0)
        # assume [0,1]; map to [-1,1]
        return (t * 2.0 - 1.0).to(device)

    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return (t * 2.0 - 1.0).to(device)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_psnr(img1, img2) -> float:
    """
    Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1, img2 : torch.Tensor (C,H,W) or np.ndarray (H,W,C)
        Ground truth and predicted image respectively.
        Can be in [0,1] float or [0,255] uint8.

    Returns
    -------
    float : PSNR in dB.  Typical good value ≥ 30 dB.
    """
    a = _to_numpy_uint8(img1)
    b = _to_numpy_uint8(img2)
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def compute_ssim(img1, img2) -> float:
    """
    Structural Similarity Index (SSIM).

    Parameters
    ----------
    img1, img2 : torch.Tensor (C,H,W) or np.ndarray (H,W,C)

    Returns
    -------
    float : SSIM score in [-1, 1].  Good value ≥ 0.85.
    """
    a = _to_numpy_uint8(img1)
    b = _to_numpy_uint8(img2)
    # channel_axis=2 tells skimage the colour axis position
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


_LPIPS_FAILED = False


def compute_lpips(img1, img2, device: torch.device = None) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) using AlexNet backbone.
    """
    global _LPIPS_NET, _LPIPS_FAILED
    if _LPIPS_FAILED or lpips_lib is None:
        return float("nan")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _LPIPS_NET is None:
        try:
            _LPIPS_NET = lpips_lib.LPIPS(net="alex").to(device)
            _LPIPS_NET.eval()
        except Exception:
            _LPIPS_FAILED = True
            return float("nan")

    t1 = _to_lpips_tensor(img1, device)
    t2 = _to_lpips_tensor(img2, device)

    try:
        with torch.no_grad():
            dist = _LPIPS_NET(t1, t2)
        return float(dist.item())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_all(
    model,
    dataloader: DataLoader,
    device: torch.device,
    use_lpips: bool = True,
) -> dict:
    """
    Run PSNR, SSIM (and optionally LPIPS) on every batch in *dataloader*
    and return mean ± std for each metric.
    """
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                damaged, clean = batch[0].to(device), batch[1].to(device)
            else:
                damaged = batch["damaged"].to(device)
                clean = batch["clean"].to(device)

            restored = model(damaged)

            for i in range(restored.size(0)):
                pred = restored[i]
                gt = clean[i]

                psnr_vals.append(compute_psnr(gt, pred))
                ssim_vals.append(compute_ssim(gt, pred))
                if use_lpips:
                    lp = compute_lpips(gt, pred, device=device)
                    if not np.isnan(lp):
                        lpips_vals.append(lp)

    def _stats(vals):
        arr = np.array(vals) if vals else np.array([float("nan")])
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "values": vals}

    result = {
        "psnr": _stats(psnr_vals),
        "ssim": _stats(ssim_vals),
    }
    if use_lpips and lpips_vals:
        result["lpips"] = _stats(lpips_vals)

    return result


# ---------------------------------------------------------------------------
# Fréchet Distance & No-Reference Quality Metrics
# ---------------------------------------------------------------------------


def compute_frechet_distance(
    features_real: np.ndarray, features_fake: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Exact mathematical Fréchet Distance between two feature distributions:
    d^2 = ||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2*(Sigma_1 * Sigma_2)^(1/2))
    Uses scipy.linalg.sqrtm.
    """
    from scipy import linalg

    mu1 = np.mean(features_real, axis=0)
    sigma1 = np.atleast_2d(np.cov(features_real, rowvar=False))

    mu2 = np.mean(features_fake, axis=0)
    sigma2 = np.atleast_2d(np.cov(features_fake, rowvar=False))

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return max(0.0, float(fid))


def compute_sharpness_score(img) -> float:
    """
    No-Reference Image Quality Assessment via Laplacian Variance.
    Quantifies edge crispness and preservation of fine geometric strokes.
    Higher indicates sharper contours and preserved line art.
    """
    arr = _to_numpy_uint8(img)
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

