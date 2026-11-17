"""Proximal variable smoothing (PVS) for the poster model (★★).

Kume & Yamada, arXiv:2506.05974v2, Algorithm 1 with the backtracking of
Algorithm 2, applied to

    minimize_{w=(z, delta, nu)}  h(w) + g(S(w)) + phi(w)
      h(w)      = 1/2 ||G(z) + delta + nu - y||^2 + kappa/2 ||delta||^2
      S(w)      = (D(G(z) + delta), nu)                 (D: SSTV operator)
      g(p, q)   = tau ||p||_1 + lam * Psi_MCP^beta(q)   ((lam/beta)-weakly convex)
      phi       = indicator of R^k x C,  C = ([-r, r]^M)^2

Iteration n (Algorithm 1):
    F_n     = h + (mu_n g) o S,     mu_n = mu_c * n^(-1/3),  mu_c = beta / (2 lam) = (2 eta)^(-1)
    w_{n+1} = P(w_n - gamma_n grad F_n(w_n)),   P = projection onto R^k x C
              (z unchanged, delta and nu clipped entrywise to [-r, r])
Step size (Algorithm 2): gamma <- gamma_init; while the Armijo-type condition (21)
    F_n(w+) <= F_n(w_n) - c * gamma * ||(w_n - w+) / gamma||^2,   w+ = P(w_n - gamma grad F_n(w_n))
fails, gamma <- rho * gamma.  (phi = 0 at feasible points, so F_n + phi = F_n there.)

The Moreau envelopes are evaluated with the prox point detached, so autograd
returns exactly (s - prox_{mu g}(s)) / mu through the chain rule (see
structured_correction/regularizers.py).

Logged every iteration: mu_n, gamma_n, number of backtracking steps, F_n(w_n),
the gradient mapping M_{gamma_n}(w_n) = ||w_n - w_{n+1}|| / gamma_n, the gradient
mapping with the fixed step gamma_init, and the fraction of clipped entries.
"""
import math
import time

import torch

from regularizers import L1, MCP, sstv_op


class PosterObjective:
    """Model (★★): tau * SSTV is applied to the corrected estimate G(z) + delta.
    use_delta=False fixes delta = 0 (the ablation "(★★) with delta = 0")."""

    def __init__(self, y, prior, tau=0.3, kappa=2.0, lam=0.2, beta=0.4, use_delta=True):
        self.y = y
        self.prior = prior
        self.use_delta = use_delta
        self.kappa = kappa if use_delta else 0.0
        self.rx = L1(weight=tau)
        # code parametrisation of MCP: lam*|t| - t^2/(2*gamma) for |t| <= gamma*lam,
        # i.e. lam * Psi^beta with beta = gamma * lam  ->  gamma = beta / lam
        self.reps = MCP(lam=lam, gamma=beta / lam)
        self.lam, self.beta = lam, beta

    def decode(self, z):
        return self.prior.decode(z)[0]

    def smoothed(self, z, delta, nu, mu, x=None):
        if x is None:
            x = self.decode(z)
        val = 0.5 * ((x + delta + nu - self.y) ** 2).sum()
        if self.kappa > 0:
            val = val + 0.5 * self.kappa * (delta ** 2).sum()
        val = val + self.rx.envelope(sstv_op(x + delta), mu)
        return val + self.reps.envelope(nu, mu)

    @torch.no_grad()
    def exact(self, z, delta, nu):
        x = self.decode(z)
        val = 0.5 * ((x + delta + nu - self.y) ** 2).sum()
        if self.kappa > 0:
            val = val + 0.5 * self.kappa * (delta ** 2).sum()
        val = val + self.rx.value(sstv_op(x + delta)) + self.reps.value(nu)
        return val.item()


def pvs(obj, z0, max_iter=600, r=2.0, gamma_init=1.0, rho=0.5, c=1e-4, mu_c=None,
        gt=None, log_every=25, gamma_min=1e-14, verbose=True):
    """Returns (x_hat clipped to [0, 1] for evaluation, components, history)."""
    from metrics import mpsnr

    if mu_c is None:
        mu_c = obj.beta / (2.0 * obj.lam)          # (2 eta)^(-1), eta = lam / beta
    z = z0.detach().clone()
    delta = torch.zeros_like(obj.y)
    nu = torch.zeros_like(obj.y)
    hist = {k: [] for k in ("mu", "gamma", "bt", "F", "M_gamma_n", "M_gamma_init",
                            "clip_delta", "clip_nu", "time")}
    hist.update(n_log=[], obj_exact=[], psnr=[], psnr_G=[])
    t0 = time.time()

    def proj(t):
        return t.clamp(-r, r)

    for n in range(1, max_iter + 1):
        mu = mu_c * n ** (-1.0 / 3.0)
        zr = z.clone().requires_grad_(True)
        nr = nu.clone().requires_grad_(True)
        dr = delta.clone().requires_grad_(obj.use_delta)
        val = obj.smoothed(zr, dr, nr, mu)
        if obj.use_delta:
            gz, gd, gn = torch.autograd.grad(val, [zr, dr, nr])
        else:
            gz, gn = torch.autograd.grad(val, [zr, nr])
            gd = torch.zeros_like(delta)
        F0 = val.item()
        with torch.no_grad():
            # gradient mapping with the fixed step gamma_init (no extra decoder call)
            di = proj(delta - gamma_init * gd) if obj.use_delta else delta
            m_init = math.sqrt(float(((gamma_init * gz) ** 2).sum() + ((delta - di) ** 2).sum()
                                     + ((nu - proj(nu - gamma_init * gn)) ** 2).sum())) / gamma_init
            gamma, bt = gamma_init, 0
            while True:
                z1 = z - gamma * gz
                d1 = proj(delta - gamma * gd) if obj.use_delta else delta
                n1 = proj(nu - gamma * gn)
                diff2 = float(((z - z1) ** 2).sum() + ((delta - d1) ** 2).sum() + ((nu - n1) ** 2).sum())
                F1 = obj.smoothed(z1, d1, n1, mu).item()
                if F1 <= F0 - c * diff2 / gamma or gamma < gamma_min:
                    break
                gamma *= rho
                bt += 1
            if gamma < gamma_min:
                print(f"  WARNING: step size fell below {gamma_min:g} at n = {n}", flush=True)
            hist["mu"].append(mu)
            hist["gamma"].append(gamma)
            hist["bt"].append(bt)
            hist["F"].append(F0)
            hist["M_gamma_n"].append(math.sqrt(diff2) / gamma)
            hist["M_gamma_init"].append(m_init)
            hist["clip_delta"].append(float((d1.abs() >= r).float().mean()) if obj.use_delta else 0.0)
            hist["clip_nu"].append(float((n1.abs() >= r).float().mean()))
            hist["time"].append(time.time() - t0)
            z, delta, nu = z1, d1, n1
            if n % log_every == 0 or n == 1 or n == max_iter:
                xg = obj.decode(z)
                xh = (xg + delta).clamp(0, 1)
                hist["n_log"].append(n)
                hist["obj_exact"].append(obj.exact(z, delta, nu))
                p = mpsnr(gt, xh) if gt is not None else float("nan")
                pg = mpsnr(gt, xg.clamp(0, 1)) if gt is not None else float("nan")
                hist["psnr"].append(p)
                hist["psnr_G"].append(pg)
                if verbose:
                    print(f"  n {n:4d}  obj {hist['obj_exact'][-1]:10.1f}  mu {mu:.3f}  gamma {gamma:.2e}"
                          f"  bt {bt:2d}  M {hist['M_gamma_n'][-1]:.3e}  MPSNR {p:.2f} (G {pg:.2f})"
                          f"  {hist['time'][-1]:.0f}s", flush=True)
    with torch.no_grad():
        xg = obj.decode(z)
        x_hat = (xg + delta).clamp(0, 1)
        comps = {"G": xg.detach(), "delta": delta.detach(), "nu": nu.detach(), "z": z.detach()}
    return x_hat, comps, hist
