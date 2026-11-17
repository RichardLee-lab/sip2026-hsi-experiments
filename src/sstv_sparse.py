"""SSTV with a sparse noise term, as in the original paper
(Aggarwal & Majumdar, IEEE GRSL 2016, eq. (4)):

    min_{X,S}  ||Y - X - S||_F^2 + lam * ||S||_1
               + mu * ( ||D_h D_s X||_1 + ||D_v D_s X||_1 ).

The paper solves it by split Bregman; we solve the same convex problem
exactly with a primal-dual method. For fixed X the minimizer in S is
S = soft(Y - X, lam / 2), so eliminating S leaves

    min_X  phi(Y - X) + mu * ||K X||_1,   phi(R) = min_S ||R - S||^2 + lam ||S||_1,

where phi is a Huber function with a 2-Lipschitz gradient
grad phi(R) = 2 (R - soft(R, lam / 2)). This is solved by the Condat-Vu
iteration with the SSTV operator K = (D_h D_s, D_v D_s) of sstv.py
(||K||^2 <= 32); step sizes satisfy tau * (L / 2 + sigma * ||K||^2) < 1
with L = 2. tau = 0.05, sigma = 0.58 converge fastest among the pairs we
tested (relative objective change < 1e-4 after 2000 iterations).

The earlier baseline in sstv.py omits S (Gaussian data term only) and is
kept for reference.
"""
import torch

from sstv import K, K_t


def _soft(r, t):
    return torch.sign(r) * torch.clamp(r.abs() - t, min=0.0)


def objective(y, x, lam, mu):
    s = _soft(y - x, lam / 2)
    k1, k2 = K(x)
    return (((y - x - s) ** 2).sum() + lam * s.abs().sum()
            + mu * (k1.abs().sum() + k2.abs().sum())).item()


def sstv_sparse_denoise(y, lam=0.1, mu=0.2, n_iter=1500, tau=0.05,
                        sigma=0.58, return_s=False, history=False):
    assert tau * (1.0 + 32.0 * sigma) < 1.0, "Condat-Vu step condition"
    x = y.clone()
    k1, k2 = K(x)
    p1, p2 = torch.zeros_like(k1), torch.zeros_like(k2)
    hist = []
    for it in range(n_iter):
        r = y - x
        grad = -2.0 * (r - _soft(r, lam / 2))
        x_new = x - tau * (grad + K_t(p1, p2, x.shape))
        k1, k2 = K(2.0 * x_new - x)
        p1 = torch.clamp(p1 + sigma * k1, -mu, mu)
        p2 = torch.clamp(p2 + sigma * k2, -mu, mu)
        x = x_new
        if history and (it + 1) % 100 == 0:
            hist.append(objective(y, x, lam, mu))
    out = x.clamp(0, 1)
    if return_s or history:
        return out, _soft(y - x, lam / 2), hist
    return out
