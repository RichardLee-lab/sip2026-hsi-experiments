"""Evaluate every saved result with the same code and write results/summary.md and results/summary.csv.

Metrics (src/metrics.py, the code used for the poster): MPSNR (up), MSSIM (up), SAM in degrees (down),
ERGAS (down), and MPSNR on the 34 real bands only. Bands 0-2 and 37-47 (0-based) of the 48-band images
are copies of the end bands (ROSIS covers 430-860 nm only), each with its own noise.

  uv run python scripts/collect_results.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import common  # noqa: E402
from metrics import evaluate, mpsnr  # noqa: E402

METHODS = [("Proposed", "Proposed (3)"), ("Proposed_delta0", "(3) with δ = 0"), ("Naive2", "(2)"),
           ("SSTV", "SSTV [2]"), ("S3TTV", "S3TTV"), ("LRTV", "LRTV"), ("LRTDTV", "LRTDTV"),
           ("HSDT", "HSDT"), ("HIRDiff", "HIR-Diff")]
SCENE_NAMES = {"PaviaU": "PaviaU", "PaviaC": "Pavia Centre"}
REAL = slice(3, 37)
# MPSNR of the same settings on the MacBook Pro (M2 Pro), 2026-09-30 to 2026-11-16, for a sanity check
MACBOOK = {("PaviaU", "Proposed"): 30.53, ("PaviaU", "Proposed_delta0"): 29.80, ("PaviaU", "SSTV"): 29.62,
           ("PaviaU", "LRTV"): 31.05, ("PaviaU", "LRTDTV"): 31.16, ("PaviaU", "HSDT"): 30.61,
           ("PaviaC", "Proposed"): 30.83, ("PaviaC", "Proposed_delta0"): 30.23, ("PaviaC", "SSTV"): 29.78,
           ("PaviaC", "LRTV"): 30.83, ("PaviaC", "LRTDTV"): 31.21, ("PaviaC", "HSDT"): 29.62}


def load(scene, method):
    """Returns (restored (B, H, W) float64, info dict) or None."""
    p = common.result_path(scene, method)
    if os.path.exists(p):
        d = np.load(p)
        return d["restored"].astype(np.float64).transpose(2, 0, 1), json.loads(str(d["info"]))
    p = os.path.join(common.RESULTS, scene, method + ".mat")              # MATLAB methods
    if os.path.exists(p):
        d = loadmat(p, squeeze_me=True, struct_as_record=False)
        mach = d["machine"]
        info = {"time_s": float(d["time_s"]),
                "machine": {"device": "cpu", "threads": int(mach.threads), "matlab": str(mach.matlab)}}
        return np.clip(d["restored"].astype(np.float64), 0, 1).transpose(2, 0, 1), info
    return None


def main():
    rows, machine, matlab = [], None, None
    for scene in common.SCENES:
        _, gt = common.load_scene(scene)
        gt = gt.numpy().astype(np.float64)
        for method, label in METHODS:
            r = load(scene, method)
            if r is None:
                rows.append(dict(scene=scene, method=method, label=label, status="not run"))
                continue
            x, info = r
            m = evaluate(gt, x)
            mach = info.get("machine", {})
            if "matlab" in mach:
                matlab = mach["matlab"]
            elif machine is None:
                machine = mach
            rows.append(dict(scene=scene, method=method, label=label, status="done",
                             MPSNR=m["MPSNR"], MSSIM=m["MSSIM"], SAM=m["SAM"], ERGAS=m["ERGAS"],
                             MPSNR_real34=mpsnr(gt[REAL], x[REAL]), time_s=info.get("time_s"),
                             device=mach.get("device"), threads=mach.get("threads"),
                             macbook_MPSNR=MACBOOK.get((scene, method))))

    lines = ["# Results on one machine", "", "Generated %s." % time.strftime("%Y-%m-%d %H:%M"), ""]
    if machine:
        lines.append("- Machine: %s, %s, %s cores, %s GB memory; Python %s, torch %s."
                     % (machine.get("model"), machine.get("cpu"), machine.get("cpu_cores"), machine.get("memory_GB"),
                        machine.get("python"), machine.get("torch")))
    if matlab:
        lines.append("- MATLAB %s (LRTV, LRTDTV)." % matlab)
    lines += ["- ↑ higher is better, ↓ lower is better. \"34 real bands\": MPSNR without the 14 copied bands.",
              "- Time: wall-clock time of the restoration itself (data and model loading excluded); "
              "`mps` = Apple GPU, `cpu` = CPU with the given number of threads.",
              "- \"MacBook\": MPSNR of the same settings on the MacBook Pro (M2 Pro), for a sanity check only.",
              "- HIR-Diff: the official sampler keeps the step with the best PSNR against the ground truth.", ""]
    for scene in common.SCENES:
        lines += ["## %s" % SCENE_NAMES[scene], "",
                  "| Method | MPSNR ↑ | MSSIM ↑ | SAM ↓ | ERGAS ↓ | MPSNR, 34 real bands ↑ | Time [s] | Device | MacBook MPSNR |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["scene"] != scene:
                continue
            if r["status"] != "done":
                lines.append("| %s | not run | | | | | | | |" % r["label"])
                continue
            dev = r["device"] + ("" if r["device"] == "mps" else " (%s threads)" % r["threads"])
            ref = "%.2f" % r["macbook_MPSNR"] if r["macbook_MPSNR"] is not None else "–"
            lines.append("| %s | %.2f | %.4f | %.2f | %.2f | %.2f | %.1f | %s | %s |"
                         % (r["label"], r["MPSNR"], r["MSSIM"], r["SAM"], r["ERGAS"], r["MPSNR_real34"],
                            r["time_s"], dev, ref))
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(common.RESULTS, exist_ok=True)
    with open(os.path.join(common.RESULTS, "summary.md"), "w") as f:
        f.write(text + "\n")
    keys = ["scene", "method", "status", "MPSNR", "MSSIM", "SAM", "ERGAS", "MPSNR_real34", "time_s", "device",
            "threads", "macbook_MPSNR"]
    with open(os.path.join(common.RESULTS, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(text)
    print("\nwritten: results/summary.md, results/summary.csv")


if __name__ == "__main__":
    main()
