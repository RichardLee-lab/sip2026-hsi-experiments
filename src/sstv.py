"""SSTV baseline (Aggarwal & Majumdar, 2016):

    min_x  1/2 ||x - y||^2 + lam * || D_hv (D_s x) ||_1

solved with the Chambolle-Pock primal-dual algorithm. D_s is the spectral
first difference, D_hv the horizontal/vertical spatial differences. All
operators and adjoints are hand-coded; the adjoint identity is unit-tested.
"""
import torch


def _ds(x):                               # (B,H,W) -> (B-1,H,W)
    return x[1:] - x[:-1]


def _ds_t(u, B):                          # adjoint of _ds
    out = torch.zeros((B,) + u.shape[1:], dtype=u.dtype, device=u.device)
    out[:-1] -= u
    out[1:] += u
    return out


def _dh(v):                               # last axis difference
    return v[..., 1:] - v[..., :-1]


def _dh_t(u, W):
    out = torch.zeros(u.shape[:-1] + (W,), dtype=u.dtype, device=u.device)
    out[..., :-1] -= u
    out[..., 1:] += u
    return out


def _dv(v):                               # row axis difference
    return v[:, 1:, :] - v[:, :-1, :]


def _dv_t(u, H):
    out = torch.zeros((u.shape[0], H, u.shape[2]), dtype=u.dtype, device=u.device)
    out[:, :-1, :] -= u
    out[:, 1:, :] += u
    return out


def K(x):
    ds = _ds(x)
    return _dh(ds), _dv(ds)


def K_t(u1, u2, shape):
    B, H, W = shape
    return _ds_t(_dh_t(u1, W), B) + _ds_t(_dv_t(u2, H), B)


def sstv_denoise(y, lam=0.05, n_iter=300, verbose=False):
    """Chambolle-Pock with ||K||^2 <= 32 (two first differences composed)."""
    tau = sigma = 1.0 / (32 ** 0.5)
    x = y.clone()
    x_bar = x.clone()
    u1, u2 = K(x)
    u1, u2 = torch.zeros_like(u1), torch.zeros_like(u2)
    for k in range(n_iter):
        k1, k2 = K(x_bar)
        u1 = torch.clamp(u1 + sigma * k1, -lam, lam)
        u2 = torch.clamp(u2 + sigma * k2, -lam, lam)
        x_new = (x - tau * K_t(u1, u2, x.shape) + tau * y) / (1 + tau)
        x_bar = 2 * x_new - x
        x = x_new
        if verbose and (k + 1) % 100 == 0:
            print(f"  sstv it {k+1}")
    return x.clamp(0, 1)
