"""Quality metrics for HSI restoration: MPSNR, MSSIM, SAM, ERGAS.

Inputs are torch tensors or numpy arrays of shape (B, H, W) in [0, 1]
(ground truth `x`, estimate `xhat`). All metrics computed in float64 numpy.
"""
import numpy as np
from skimage.metrics import structural_similarity


def _np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def mpsnr(x, xhat, data_range=1.0):
    x, xhat = _np(x), _np(xhat)
    mse = ((x - xhat) ** 2).mean(axis=(1, 2))
    mse = np.maximum(mse, 1e-12)
    return float(np.mean(10 * np.log10(data_range ** 2 / mse)))


def mssim(x, xhat, data_range=1.0):
    x, xhat = _np(x), _np(xhat)
    vals = [structural_similarity(x[b], xhat[b], data_range=data_range)
            for b in range(x.shape[0])]
    return float(np.mean(vals))


def sam(x, xhat, eps=1e-8):
    """Mean spectral angle mapper over pixels, in degrees."""
    x, xhat = _np(x), _np(xhat)
    B = x.shape[0]
    u = x.reshape(B, -1)
    v = xhat.reshape(B, -1)
    dot = (u * v).sum(0)
    cos = dot / (np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0) + eps)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(ang.mean())


def ergas(x, xhat, ratio=1.0):
    """ERGAS with resolution ratio 1 (denoising setting)."""
    x, xhat = _np(x), _np(xhat)
    B = x.shape[0]
    rmse2 = ((x - xhat) ** 2).mean(axis=(1, 2))
    mean2 = np.maximum(x.mean(axis=(1, 2)) ** 2, 1e-12)
    return float(100 * ratio * np.sqrt((rmse2 / mean2).mean()))


def evaluate(x, xhat):
    return {"MPSNR": mpsnr(x, xhat), "MSSIM": mssim(x, xhat),
            "SAM": sam(x, xhat), "ERGAS": ergas(x, xhat)}
