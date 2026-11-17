"""SSTV [2] with its sparse-noise term (Aggarwal & Majumdar, IEEE GRSL 2016, eq. (4)), src/sstv_sparse.py:

    min_{X,S} ||Y - X - S||_F^2 + lam ||S||_1 + mu (||D_h D_s X||_1 + ||D_v D_s X||_1),

solved by a primal-dual (Condat-Vu) iteration, 1500 iterations. lam = 0.2, mu = 0.4 were chosen on
PaviaU by the 2026-09-30 grid (36 pairs) and are used for both scenes.

  uv run python scripts/run_sstv.py --scene PaviaU
  uv run python scripts/run_sstv.py --scene PaviaU --test     # 50 iterations, nothing saved
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import common  # noqa: E402
from metrics import evaluate  # noqa: E402
from sstv_sparse import sstv_sparse_denoise  # noqa: E402

LAM, MU, ITERS = 0.2, 0.4, 1500
METHOD = "SSTV"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=common.SCENES, required=True)
    ap.add_argument("--device", default="auto", help="auto (= mps if available), mps or cpu")
    ap.add_argument("--threads", type=int, default=common.default_threads())
    ap.add_argument("--test", action="store_true", help="50 iterations only, nothing saved")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if common.done(a.scene, METHOD) and not (a.test or a.force):
        print("skip %s %s: result exists (use --force to run again)" % (METHOD, a.scene))
        return
    torch.set_num_threads(a.threads)
    dev = common.pick_device(a.device)
    iters = 50 if a.test else ITERS
    y, gt = common.load_scene(a.scene)
    yd = y.to(dev)
    common.sync(dev)
    t0 = time.time()
    x = sstv_sparse_denoise(yd, lam=LAM, mu=MU, n_iter=iters)
    common.sync(dev)
    dt = time.time() - t0
    x = x.cpu()
    m = evaluate(gt, x)
    print("%s %s: MPSNR %.2f  MSSIM %.4f  SAM %.2f  (%.1f s, device %s)"
          % (METHOD, a.scene, m["MPSNR"], m["MSSIM"], m["SAM"], dt, dev), flush=True)
    if a.test:
        print("TEST OK", flush=True)
        return
    out = common.save_result(a.scene, METHOD, x, dt, dict(lam=LAM, mu=MU, iterations=iters, tau=0.05, sigma=0.58),
                             dev, a.threads)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
