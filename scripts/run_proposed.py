"""Proposed model (3) and its two ablations, solved by proximal variable smoothing with backtracking
(src/solver_pvs.py; the same code and settings as the poster runs of 2026-11-10/11).

  variant    saved as          model
  proposed   Proposed          (3): tau * SSTV on G(z) + delta, kappa = 2
  delta0     Proposed_delta0   (3) with delta fixed to 0
  naive2     Naive2            (2): delta = 0 and tau = 0 (only the MCP term on nu)

Common settings: lam = 0.2, beta = 0.4 (MCP on nu), r = 2 (box on delta and nu), 600 iterations,
gamma_init = 0.031 (chosen once by the PaviaU pilot of 2026-11-10 and fixed for every run),
Armijo rho = 0.5, c = 1e-4, mu_n = beta / (2 lam) * n^(-1/3). The ground truth is used only to print
the MPSNR every 100 iterations; it never enters the iteration.

  uv run python scripts/run_proposed.py --scene PaviaU --variant proposed
  uv run python scripts/run_proposed.py --scene PaviaU --variant proposed --test   # 5 iterations, nothing saved
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import common  # noqa: E402
from metrics import evaluate, mpsnr  # noqa: E402
from solver_pvs import PosterObjective, pvs  # noqa: E402

TAU, KAPPA, LAM, BETA, R, ITERS = 0.3, 2.0, 0.2, 0.4, 2.0, 600
GAMMA_INIT, RHO, C_ARMIJO = 0.031, 0.5, 1e-4
VARIANTS = {"proposed": ("Proposed", dict(use_delta=True, kappa=KAPPA, tau=TAU)),
            "delta0": ("Proposed_delta0", dict(use_delta=False, kappa=0.0, tau=TAU)),
            "naive2": ("Naive2", dict(use_delta=False, kappa=0.0, tau=0.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=common.SCENES, required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--device", default="auto", help="auto (= mps if available), mps or cpu")
    ap.add_argument("--threads", type=int, default=common.default_threads())
    ap.add_argument("--test", action="store_true", help="5 iterations only, nothing saved")
    ap.add_argument("--force", action="store_true", help="run again even if the result exists")
    a = ap.parse_args()
    method, cfg = VARIANTS[a.variant]
    if common.done(a.scene, method) and not (a.test or a.force):
        print("skip %s %s: results/%s/%s.npz exists (use --force to run again)" % (method, a.scene, a.scene, method))
        return
    torch.set_num_threads(a.threads)
    dev = common.pick_device(a.device)
    iters = 5 if a.test else a.iters
    print("%s on %s: %s, lam %g, beta %g, r %g, %d iterations, gamma_init %g, device %s"
          % (method, a.scene, cfg, LAM, BETA, R, iters, GAMMA_INIT, dev), flush=True)

    from vae_prior import HSIGenePrior
    prior = HSIGenePrior(device=dev)
    y, gt = common.load_scene(a.scene)
    yd, gtd = y.to(dev), gt.to(dev)

    common.sync(dev)
    t0 = time.time()
    with torch.no_grad():
        z0 = prior.encode(yd.clamp(0, 1).unsqueeze(0))
    obj = PosterObjective(yd, prior, tau=cfg["tau"], kappa=cfg["kappa"], lam=LAM, beta=BETA,
                          use_delta=cfg["use_delta"])
    xh, comps, hist = pvs(obj, z0, max_iter=iters, r=R, gamma_init=GAMMA_INIT, rho=RHO, c=C_ARMIJO,
                          gt=gtd, log_every=1 if a.test else 100)
    common.sync(dev)
    dt = time.time() - t0
    xh = xh.detach().cpu()
    m = evaluate(gt, xh)
    print("%s %s: MPSNR %.2f  MSSIM %.4f  SAM %.2f  (%.0f s, %.2f s/iteration, mean backtracks %.2f)"
          % (method, a.scene, m["MPSNR"], m["MSSIM"], m["SAM"], dt, dt / iters, np.mean(hist["bt"])), flush=True)
    if a.test:
        print("time for %d iterations: at most about %.1f min (the first iterations are slower)"
              % (ITERS, ITERS * dt / iters / 60), flush=True)
        print("TEST OK", flush=True)
        return
    params = dict(cfg, lam=LAM, beta=BETA, r=R, iterations=iters, gamma_init=GAMMA_INIT, rho=RHO, c=C_ARMIJO)
    comps = {k: v.detach().cpu() for k, v in comps.items()}
    out = common.save_result(a.scene, method, xh, dt, params, dev, a.threads,
                             mean_backtracks=round(float(np.mean(hist["bt"])), 2),
                             mpsnr_G_only=round(float(mpsnr(gt, comps["G"].clamp(0, 1))), 4),
                             delta_norm=round(float(comps["delta"].norm()), 3),
                             nu_norm=round(float(comps["nu"].norm()), 3))
    np.savez_compressed(os.path.join(common.RESULTS, a.scene, "%s_parts.npz" % method),
                        **{k: comps[k].numpy().astype("float32") for k in ("G", "delta", "nu")})
    common.save_json(a.scene, "%s_hist.json" % method, hist)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
