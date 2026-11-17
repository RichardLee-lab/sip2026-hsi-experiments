"""S3TTV (Takemoto, Naganuma, Ono, IEEE JSTARS 2025), official Python/PyTorch port (S3TTV_fast, CPU).

Settings: the official defaults (fast P-PDS, stopping criterion 1e-5, at most 20000 iterations, SVT by
eigh), except
- block size 8 x 8 instead of 10 x 10: our images are 256 x 256 and 256 is not divisible by 10 (the
  official prox leaves the pixels outside the last complete block unregularized);
- radii = 0.95 x the true noise norms (configs/s3ttv_radii.json), i.e. the noise levels are known, as in
  the paper.

  uv run python scripts/run_s3ttv.py --scene PaviaU
  uv run python scripts/run_s3ttv.py --scene PaviaU --test      # 30 iterations: speed and time estimate
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import common  # noqa: E402
from metrics import evaluate  # noqa: E402

sys.path.insert(0, os.path.join(common.REPO, "external", "S3TTV"))
from solvers_python.S3TTV_fast_python import S3TTV_fast  # noqa: E402

METHOD = "S3TTV"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=common.SCENES, required=True)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--stopcri", type=float, default=1e-5)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--threads", type=int, default=common.default_threads())
    ap.add_argument("--test", action="store_true", help="30 iterations only, nothing saved")
    ap.add_argument("--crop", type=int, default=0, help="(debugging) use the top-left crop of this size")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if common.done(a.scene, METHOD) and not (a.test or a.force):
        print("skip %s %s: result exists (use --force to run again)" % (METHOD, a.scene))
        return
    torch.set_num_threads(a.threads)
    y, gt = common.load_scene(a.scene)                       # (B, H, W)
    if a.crop:
        y, gt = y[:, :a.crop, :a.crop], gt[:, :a.crop, :a.crop]
    v = y.permute(1, 2, 0).numpy().astype(np.float32)        # (H, W, B), as the official code expects
    r = dict(json.load(open(os.path.join(common.REPO, "configs", "s3ttv_radii.json")))[a.scene])
    if a.crop:   # rough rescaling of the radii to the crop (l2 norm ~ sqrt(#pixels), l1 norms ~ #pixels)
        f = (a.crop / 256.0) ** 2
        r = dict(epsilon=r["epsilon"] * f ** 0.5, alpha=r["alpha"] * f, beta=r["beta"] * f)
    p = {"epsilon": r["epsilon"], "alpha": r["alpha"], "beta": r["beta"], "blocksize": [a.block, a.block],
         "maxiter": 30 if a.test else a.maxiter, "stopcri": a.stopcri, "disprate": 10 if a.test else 500,
         "svt_method": "eigh"}
    print("%s %s: size %s, %d threads, params %s" % (METHOD, a.scene, v.shape, a.threads, p), flush=True)
    t0 = time.time()
    x, _, it, conv = S3TTV_fast(v, p, device="cpu")
    dt = time.time() - t0
    x = torch.from_numpy(np.clip(np.asarray(x, dtype=np.float32), 0, 1)).permute(2, 0, 1)
    m = evaluate(gt, x)
    print("%s %s: %d iterations, %.0f s (%.2f s/iteration), MPSNR %.2f  MSSIM %.4f  SAM %.2f, last change %.2e"
          % (METHOD, a.scene, it, dt, dt / it, m["MPSNR"], m["MSSIM"], m["SAM"], conv[-1]), flush=True)
    if a.test:
        print("estimated time: 2000 it. %.1f h / 5000 it. %.1f h / 20000 it. (upper limit) %.1f h"
              % (2000 * dt / it / 3600, 5000 * dt / it / 3600, 20000 * dt / it / 3600), flush=True)
        print("TEST OK", flush=True)
        return
    if a.crop:
        print("crop runs are not saved")
        return
    out = common.save_result(a.scene, METHOD, x, dt, dict(p, iterations=int(it)), "cpu", a.threads,
                             iterations=int(it), last_relative_change=float(conv[-1]))
    common.save_json(a.scene, "%s_hist.json" % METHOD, {"converge_rate_U": [float(c) for c in conv]})
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
