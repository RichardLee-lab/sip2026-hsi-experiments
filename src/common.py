"""Shared helpers for the Python runners: paths, data loading, devices, machine info, saving results.

Every runner saves results/<scene>/<method>.npz with
    restored : float32 array (H, W, B), the restored image clipped to [0, 1]
    info     : JSON string (method, scene, parameters, run time, machine)
and a copy of `info` as results/<scene>/<method>.json, so that collect_results.py can evaluate
all methods (including the MATLAB ones, saved as .mat) in one place.
"""
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
from scipy.io import loadmat

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(REPO, "results")
SCENES = ("PaviaU", "PaviaC")


def data_path(scene):
    return os.path.join(DATA, "%s_case5_seed0.mat" % scene)


def load_scene(scene):
    """Noisy input y and ground truth gt as float32 tensors of shape (B, H, W) = (48, 256, 256)."""
    d = loadmat(data_path(scene))
    y = torch.from_numpy(d["input"]).permute(2, 0, 1).float().contiguous()
    gt = torch.from_numpy(d["gt"]).permute(2, 0, 1).float().contiguous()
    return y, gt


def default_threads():
    """Half of the CPU cores: the Mac Studio is shared, and MATLAB uses the same number."""
    return max(1, (os.cpu_count() or 2) // 2)


def pick_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS (Apple GPU) is not available here; run again with --device cpu")
    return name


def sync(device):
    """Wait for the GPU before reading the clock."""
    if device == "mps":
        torch.mps.synchronize()


def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def machine_info(device=None, threads=None):
    mem = _sysctl("hw.memsize")
    return {"model": _sysctl("hw.model"),
            "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
            "cpu_cores": os.cpu_count(),
            "memory_GB": round(int(mem) / 2 ** 30, 1) if mem.isdigit() else None,
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "threads": threads}


def result_path(scene, method):
    return os.path.join(RESULTS, scene, "%s.npz" % method)


def done(scene, method):
    return os.path.exists(result_path(scene, method))


def save_result(scene, method, x, time_s, params, device, threads, **extra):
    """x: restored image of shape (B, H, W), torch or numpy. Stored as (H, W, B) float32 in [0, 1]."""
    os.makedirs(os.path.join(RESULTS, scene), exist_ok=True)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0).transpose(1, 2, 0)
    info = dict(method=method, scene=scene, time_s=round(float(time_s), 2), params=params,
                machine=machine_info(device, threads), finished=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    with open(os.path.join(RESULTS, scene, "%s.json" % method), "w") as f:
        json.dump(info, f, indent=1)
    # write to a temporary file first, so that an interrupted save never looks like a finished run
    path = result_path(scene, method)
    tmp = path[:-4] + ".partial.npz"
    np.savez_compressed(tmp, restored=x, info=json.dumps(info))
    os.replace(tmp, path)
    return path


def save_json(scene, name, obj):
    os.makedirs(os.path.join(RESULTS, scene), exist_ok=True)
    with open(os.path.join(RESULTS, scene, name), "w") as f:
        json.dump(obj, f)
