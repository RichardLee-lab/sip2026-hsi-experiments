"""HSDT (Lai et al., ICCV 2023), the released "l" model trained on ICVL with complex noise
(weights/hsdt_l_complex.pth), applied to our 48-band images without retraining.

HSDT's GSSA attention has a reshape fixed to 31 bands; we use its band-count-agnostic "flex" form
(the data-dependent attention that HSDT also uses during training), as in our earlier runs.

  uv run python scripts/run_hsdt.py --scene PaviaU
  uv run python scripts/run_hsdt.py --scene PaviaU --test     # runs the network, nothing saved
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "external", "HSDT"))
import common  # noqa: E402
from metrics import evaluate  # noqa: E402

WEIGHTS = os.path.join(common.REPO, "weights", "hsdt_l_complex.pth")
METHOD = "HSDT"


def _gssa_flex_forward(self, x):
    """GSSA flex-mode forward with a reshape that works for any number of bands."""
    B, C, D, H, W = x.shape
    residual = x
    tmp = x.reshape(B, C, D, H * W).mean(-1).permute(0, 2, 1)
    attn = tmp @ tmp.transpose(1, 2)                      # (B, D, D), any D
    attn = F.softmax(attn, dim=-1).unsqueeze(1).unsqueeze(1)
    v = self.value_proj(rearrange(x, "b c d h w -> b h w d c"))
    q = self.fc(torch.matmul(attn, v))
    q = rearrange(q, "b h w d c -> b c d h w") + residual
    return q, attn


def build(device):
    from hsdt.arch import HSDT
    from hsdt.attention import GSSA
    net = HSDT(1, 16, 7, [1, 3, 5])                       # the released "l" variant
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False), strict=True)
    for m in net.modules():
        if type(m) is GSSA:
            m.forward = _gssa_flex_forward.__get__(m, GSSA)
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=common.SCENES, required=True)
    ap.add_argument("--device", default="auto", help="auto (= mps if available), mps or cpu")
    ap.add_argument("--threads", type=int, default=common.default_threads())
    ap.add_argument("--test", action="store_true", help="run but do not save")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if common.done(a.scene, METHOD) and not (a.test or a.force):
        print("skip %s %s: result exists (use --force to run again)" % (METHOD, a.scene))
        return
    torch.set_num_threads(a.threads)
    dev = common.pick_device(a.device)
    net = build(dev)
    y, gt = common.load_scene(a.scene)
    inp = y.clamp(0, 1).unsqueeze(0).unsqueeze(0).to(dev)  # (1, 1, B, H, W)
    with torch.no_grad():                                  # warm-up on a small crop (GPU kernels), not timed
        net(inp[..., :64, :64])
    common.sync(dev)
    t0 = time.time()
    with torch.no_grad():
        x = net(inp)
    common.sync(dev)
    dt = time.time() - t0
    x = x[0, 0].cpu().clamp(0, 1)
    m = evaluate(gt, x)
    print("%s %s: MPSNR %.2f  MSSIM %.4f  SAM %.2f  (%.1f s, device %s)"
          % (METHOD, a.scene, m["MPSNR"], m["MSSIM"], m["SAM"], dt, dev), flush=True)
    if a.test:
        print("TEST OK", flush=True)
        return
    out = common.save_result(a.scene, METHOD, x, dt, dict(weights="hsdt_l_complex.pth", gssa="flex"), dev, a.threads)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
