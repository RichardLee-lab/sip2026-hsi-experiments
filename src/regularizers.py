"""Weakly convex regularizers, their proximity operators and Moreau envelopes,
plus the SSTV linear operator.

Moreau envelope of g with parameter mu:
    (mu)g(s) = min_u g(u) + ||u - s||^2 / (2 mu),
whose gradient is  (s - prox_{mu g}(s)) / mu.

`envelope()` computes the envelope VALUE with the prox point detached from the
autograd graph. Differentiating that value w.r.t. s then yields exactly
(s - prox)/mu, so the whole smoothed objective can be optimized by autograd
without hand-coding adjoints (the chain rule through G_theta and the linear
operator S is handled by torch).
"""
import torch


class L1:
    """g(u) = w * ||u||_1  (convex => 0-weakly convex)."""

    def __init__(self, weight=1.0):
        self.w = weight

    def value(self, u):
        return self.w * u.abs().sum()

    def prox(self, s, mu):
        t = mu * self.w
        return torch.sign(s) * torch.clamp(s.abs() - t, min=0.0)

    def envelope(self, s, mu):
        u = self.prox(s.detach(), mu)
        return self.value(u) + ((s - u) ** 2).sum() / (2 * mu)


class MCP:
    """Minimax concave penalty (Zhang, 2010), elementwise:

        rho(t) = lam*|t| - t^2/(2*gamma)   if |t| <= gamma*lam
                 gamma*lam^2/2             otherwise.

    rho is (1/gamma)-weakly convex; prox_{mu*rho} is the firm-thresholding
    operator, closed form for mu < gamma.
    """

    def __init__(self, lam=1.0, gamma=4.0):
        assert gamma > 0
        self.lam = lam
        self.gamma = gamma

    def value(self, u):
        a = u.abs()
        inner = self.lam * a - u ** 2 / (2 * self.gamma)
        outer = torch.full_like(u, self.gamma * self.lam ** 2 / 2)
        return torch.where(a <= self.gamma * self.lam, inner, outer).sum()

    def prox(self, s, mu):
        assert mu < self.gamma, f"prox needs mu < gamma (mu={mu}, gamma={self.gamma})"
        a = s.abs()
        soft = torch.sign(s) * torch.clamp(a - mu * self.lam, min=0.0) / (1 - mu / self.gamma)
        out = torch.where(a <= self.gamma * self.lam, soft, s)
        return torch.where(a <= mu * self.lam, torch.zeros_like(s), out)

    def envelope(self, s, mu):
        u = self.prox(s.detach(), mu)
        return self.value(u) + ((s - u) ** 2).sum() / (2 * mu)


class Nuclear:
    """g(U) = w * ||U||_*  on a matrix (convex). prox = singular value
    thresholding. SVD runs on CPU (MPS lacks linalg.svd); that is fine
    because the envelope uses a detached prox point — no grad through SVD."""

    def __init__(self, weight=1.0):
        self.w = weight

    def value(self, u):
        s = torch.linalg.svdvals(u.detach().cpu())
        return (self.w * s.sum()).to(u.device)

    def prox(self, s_mat, mu):
        dev = s_mat.device
        U, s, Vh = torch.linalg.svd(s_mat.detach().cpu(), full_matrices=False)
        s = torch.clamp(s - mu * self.w, min=0.0)
        r = int((s > 0).sum())
        return ((U[:, :r] * s[:r]) @ Vh[:r]).to(dev)

    def envelope(self, s_mat, mu):
        u = self.prox(s_mat, mu)
        # value(u) recomputes an SVD; use the thresholded singular values
        # implicitly via prox: value is only needed up to a constant in the
        # gradient, but we return the exact envelope for monitoring.
        return self.value(u) + ((s_mat - u) ** 2).sum() / (2 * mu)


def unfold_spectral(x):
    """(B, H, W) -> casewise unfolding (HW, B) for low-rank penalties."""
    return x.reshape(x.shape[0], -1).T


def tv_op(x):
    """Band-wise anisotropic spatial TV features: stacked D_h x, D_v x."""
    dh = x[:, :, 1:] - x[:, :, :-1]
    dv = x[:, 1:, :] - x[:, :-1, :]
    return torch.cat([dh.reshape(-1), dv.reshape(-1)])


def sstv_op(x):
    """SSTV feature map: spatial gradients of the spectral difference cube.

    x: (B, H, W). Returns a single flat tensor stacking D_h(D_s x), D_v(D_s x)
    so an l1-type penalty on it realizes SSTV (Aggarwal & Majumdar, 2016).
    """
    ds = x[1:] - x[:-1]                       # spectral difference (B-1, H, W)
    dh = ds[:, :, 1:] - ds[:, :, :-1]         # horizontal
    dv = ds[:, 1:, :] - ds[:, :-1, :]         # vertical
    return torch.cat([dh.reshape(-1), dv.reshape(-1)])
