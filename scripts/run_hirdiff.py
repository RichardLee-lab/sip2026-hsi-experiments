"""HIR-Diff (Pang et al., CVPR 2024) with the official denoising recipe: exp schedule, k = 8, 20 steps,
eta1 = 16, eta2 = 10, Rr = 3; equal-interval band selection (the official sRRQR needs the MATLAB engine);
random seed 0.

Protocol note (state it in the paper): the official sampler compares every step with the ground truth and
returns the best step if the last one is worse. We keep this official behaviour.

Needs the pretrained diffusion model of ddpm-cd (1.56 GB, too large for GitHub):
    external/HIRDiff/checkpoints/diffusion/I190000_E97_gen.pth
Without it the script exits with code 3 ("skipped").

  uv run python scripts/run_hirdiff.py --scene PaviaU
  uv run python scripts/run_hirdiff.py --scene PaviaU --test     # 2 diffusion steps, nothing saved
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # a few operators are not on the Apple GPU

import argparse  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")                     # the official sampler calls plt.show()
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import common  # noqa: E402
from metrics import evaluate  # noqa: E402

HIRDIFF_DIR = os.path.join(common.REPO, "external", "HIRDiff")
sys.path.insert(0, HIRDIFF_DIR)
CKPT = os.path.join(HIRDIFF_DIR, "checkpoints", "diffusion", "I190000_E97_gen.pth")
METHOD = "HIRDiff"
ETA1, ETA2, STEP, RR, SEED = 16.0, 10.0, 20, 3, 0


def build(device):
    import json
    from guided_diffusion.create import create_model_and_diffusion_RS
    with open(os.path.join(HIRDIFF_DIR, "configs", "base.json")) as f:
        opt = json.load(f)
    opt.update({"beta_schedule": "exp", "beta_linear_start": 1e-6, "beta_linear_end": 1e-2,
                "cosine_s": 0, "k": 8})
    model, diffusion = create_model_and_diffusion_RS(opt)
    cks = th.load(CKPT, map_location="cpu")
    model.load_state_dict(OrderedDict((k[11:] if k.startswith("denoise_fn.") else k, v) for k, v in cks.items()),
                          strict=False)
    return model.to(device).eval(), diffusion


def restore(y, gt, model, diffusion, device, step):
    """y, gt: (B, H, W) tensors. Returns the restored (B, H, W) tensor on the CPU."""
    from utility import cal_bwpsnr
    gt = gt.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    _, ch, h, w = gt.shape
    param = {"task": "denoise", "eta1": ETA1 * 256 * 64, "eta2": ETA2 * 8 * 64}
    cond = {"input": y, "gt": gt, "sigma": 0.0, "transform": lambda x: x}
    param["Band"] = th.tensor([ch * i // (RR + 1) for i in range(1, RR + 1)], dtype=th.int).to(device)
    diffusion.best_psnr, diffusion.best_result = 0, None
    sample, e_out = diffusion.p_sample_loop(model, (1, ch, h, w), Rr=RR, step=step, clip_denoised=True,
                                            denoised_fn={"denoise_model": None, "denoise_optim": None},
                                            model_condition=cond, param=param, save_root=None, progress=True)
    sample = (sample + 1) / 2
    out = th.clip(th.matmul(e_out, sample.reshape(1, RR, -1)).reshape(1, ch, h, w), 0, 1)
    if np.mean(cal_bwpsnr(out, gt)) < diffusion.best_psnr and diffusion.best_result is not None:
        out = diffusion.best_result                       # official behaviour (see the note above)
    return out[0].detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=common.SCENES, required=True)
    ap.add_argument("--device", default="auto", help="auto (= mps if available), mps or cpu")
    ap.add_argument("--threads", type=int, default=common.default_threads())
    ap.add_argument("--test", action="store_true", help="2 diffusion steps, nothing saved")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if common.done(a.scene, METHOD) and not (a.test or a.force):
        print("skip %s %s: result exists (use --force to run again)" % (METHOD, a.scene))
        return
    if not os.path.exists(CKPT):
        print("SKIPPED %s: the pretrained model %s is missing (1.56 GB, not on GitHub; see README)"
              % (METHOD, os.path.relpath(CKPT, common.REPO)), flush=True)
        sys.exit(3)
    th.set_num_threads(a.threads)
    dev = common.pick_device(a.device)
    step = 2 if a.test else STEP
    model, diffusion = build(dev)
    y, gt = common.load_scene(a.scene)
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    common.sync(dev)
    t0 = time.time()
    x = restore(y, gt, model, diffusion, dev, step)
    common.sync(dev)
    dt = time.time() - t0
    m = evaluate(gt, x)
    print("%s %s: MPSNR %.2f  MSSIM %.4f  SAM %.2f  (%.1f s, %d steps, device %s)"
          % (METHOD, a.scene, m["MPSNR"], m["MSSIM"], m["SAM"], dt, step, dev), flush=True)
    if a.test:
        print("TEST OK", flush=True)
        return
    out = common.save_result(a.scene, METHOD, x, dt,
                             dict(eta1=ETA1, eta2=ETA2, step=step, Rr=RR, bands="equal-interval", seed=SEED,
                                  best_step_by_gt="official behaviour"), dev, a.threads)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
