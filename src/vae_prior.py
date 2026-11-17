"""HSIGene pretrained VAE wrapper used as the generative prior G_theta.

The decoder maps a latent z (4 x H/4 x W/4) to an HSI (48 x H x W).
Weights are frozen; only z is optimized at restoration time.

Conventions:
  - HSIs in experiment code live in [0, 1], shape (48, H, W), float32.
  - The VAE internally works in [-1, 1] (LDM convention, cf. inference_uncond.py).
  - z follows the diffusion latent scaling: z = scale_factor * encoder_output,
    decode applies 1/scale_factor first. scale_factor = 0.18215.
"""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

HSIGENE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "HSIGene"))
SCALE_FACTOR = 0.18215


def _import_autoencoder():
    if HSIGENE_DIR not in sys.path:
        sys.path.insert(0, HSIGENE_DIR)
    from ldm.models.autoencoder import AutoencoderKL  # noqa: E402
    return AutoencoderKL


def load_vae(ckpt_path=None, device="cpu"):
    """Instantiate AutoencoderKL from the HSIGene inference config and load
    the first-stage weights from the full checkpoint (or a pre-extracted
    vae-only checkpoint)."""
    AutoencoderKL = _import_autoencoder()
    cfg = OmegaConf.load(os.path.join(HSIGENE_DIR, "configs", "inference.yaml"))
    fs = cfg.model.params.first_stage_config.params
    vae = AutoencoderKL(ddconfig=OmegaConf.to_container(fs.ddconfig),
                        lossconfig={"target": "torch.nn.Identity"},
                        embed_dim=fs.embed_dim)

    if ckpt_path is None:
        for cand in ("vae_only.pt", "last.ckpt", "last.pth"):
            p = os.path.join(HSIGENE_DIR, "checkpoints", cand)
            if os.path.exists(p):
                ckpt_path = p
                break
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found in HSIGene/checkpoints")

    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    prefix = "first_stage_model."
    vae_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not vae_sd:  # already a vae-only checkpoint
        vae_sd = sd
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    assert not missing, f"missing VAE keys: {missing[:5]}"
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def extract_vae_only(full_ckpt, out_path):
    """Save just the first-stage weights so we never need the 6GB file again."""
    sd = torch.load(full_ckpt, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    prefix = "first_stage_model."
    vae_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    assert vae_sd, "no first_stage_model.* keys found"
    torch.save(vae_sd, out_path)
    return out_path


class HSIGenePrior(nn.Module):
    """G_theta with [0,1] input/output convention.

    decode: z (B, 4, h, w) -> x (B, 48, 4h, 4w) in [0, 1]
    encode: x in [0, 1] -> z (posterior mean, scaled) — used to initialize z.
    """

    def __init__(self, ckpt_path=None, device="cpu"):
        super().__init__()
        self.vae = load_vae(ckpt_path, device)
        self.device = device

    def decode(self, z):
        x = self.vae.decode(z / SCALE_FACTOR)
        return (x + 1.0) / 2.0

    @torch.no_grad()
    def encode(self, x01):
        posterior = self.vae.encode(x01 * 2.0 - 1.0)
        return posterior.mode() * SCALE_FACTOR

    def latent_shape(self, H, W):
        return (1, 4, H // 4, W // 4)
