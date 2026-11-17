"""Check the environment before running: machine, packages, data, weights, and that every method's code loads.
Exit code 1 if something essential is missing (the HIR-Diff model is optional: HIR-Diff is then skipped).

  uv run python scripts/check_env.py
"""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import common  # noqa: E402

ok = True


def check(name, fn, essential=True):
    global ok
    try:
        msg = fn()
        print("  OK      %-34s %s" % (name, msg or ""))
    except Exception as e:  # noqa: BLE001
        print("  %s %-34s %s: %s" % ("FAILED " if essential else "MISSING", name, type(e).__name__, e))
        if essential:
            traceback.print_exc(limit=1)
            ok = False


def main():
    import torch
    info = common.machine_info()
    print("machine: %s, %s, %s cores, %s GB memory, %s" % (info["model"], info["cpu"], info["cpu_cores"],
                                                          info["memory_GB"], info["os"]))
    print("python %s, torch %s, Apple GPU (MPS): %s, CPU threads for the runs: %d"
          % (info["python"], info["torch"], torch.backends.mps.is_available(), common.default_threads()))

    def packages():
        for p in ("numpy", "scipy", "skimage", "einops", "omegaconf", "pytorch_lightning", "matplotlib", "tqdm",
                  "imgvision"):
            importlib.import_module(p)
    check("python packages", packages)

    def data():
        shapes = []
        for s in common.SCENES:
            y, gt = common.load_scene(s)
            assert y.shape == gt.shape == (48, 256, 256), y.shape
            shapes.append("%s %s" % (s, tuple(y.shape)))
        return ", ".join(shapes)
    check("data (case 5, seed 0)", data)

    def hsigene():
        from vae_prior import HSIGenePrior
        prior = HSIGenePrior(device="cpu")
        with torch.no_grad():
            x = prior.decode(torch.zeros(1, 4, 8, 8))
        return "decoder output %s" % (tuple(x.shape),)
    check("HSIGene VAE (proposed method)", hsigene)

    def hsdt():
        import run_hsdt
        run_hsdt.build("cpu")
    sys.path.insert(0, HERE)
    check("HSDT network + weights", hsdt)

    def hirdiff_code():
        import run_hirdiff  # noqa: F401
        from guided_diffusion.create import create_model_and_diffusion_RS  # noqa: F401
        from utility import cal_bwpsnr  # noqa: F401
    check("HIR-Diff code", hirdiff_code)

    def hirdiff_ckpt():
        import run_hirdiff
        if not os.path.exists(run_hirdiff.CKPT):
            raise FileNotFoundError("external/HIRDiff/checkpoints/diffusion/I190000_E97_gen.pth "
                                    "(HIR-Diff will be skipped)")
        return "%.2f GB" % (os.path.getsize(run_hirdiff.CKPT) / 1e9)
    check("HIR-Diff pretrained model", hirdiff_ckpt, essential=False)

    def s3ttv():
        import run_s3ttv  # noqa: F401
    check("S3TTV code", s3ttv)

    print("environment check: %s" % ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
