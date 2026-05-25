"""PiD model manager — loads checkpoints and runs inference.

Handles ComfyUI ↔ PiD data format conversions:
- ComfyUI LATENT: {"samples": [B,C,H,W]} → PiD LQ_latent: [B,C,H,W]
- ComfyUI IMAGE: [B,H,W,C] float32 [0,1] → PiD image: [B,3,H,W] float32 [-1,1]
- PiD output: [B,3,H,W] [-1,1] → ComfyUI IMAGE: [B,H,W,C] [0,1]

All PiD imports are deferred to runtime to avoid crashing ComfyUI startup
when PiD dependencies (omegaconf, hydra-core, etc.) are not yet installed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}


def _ensure_pid_in_path():
    """Add PiD source tree to sys.path if present."""
    pid_dir = Path(__file__).parent.parent / "PiD"
    if pid_dir.exists() and str(pid_dir) not in sys.path:
        sys.path.insert(0, str(pid_dir))


def _setup_compat():
    """Apply Windows compatibility patches."""
    from core.compat import setup_pid_compat
    setup_pid_compat()


def _load_pid_model(
    backbone: str,
    ckpt_type: str,
    checkpoint_path: str | None = None,
) -> Any:
    """Load a PiD model from checkpoint.

    Args:
        backbone: "flux", "flux2", "sd3", "zimage", "rae", "scale_rae"
        ckpt_type: "2k" or "2kto4k"
        checkpoint_path: explicit path, or None to use registry default

    Returns:
        Loaded PiD model instance.
    """
    _ensure_pid_in_path()
    _setup_compat()

    # Deferred imports — PiD deps may not be available at plugin load time
    from pid._src.inference.checkpoint_registry import get_pid_checkpoint
    from pid._src.utils.model_loader import load_model_from_checkpoint

    ckpt_info = get_pid_checkpoint(backbone, ckpt_type)
    experiment = ckpt_info.experiment
    ckpt_path = checkpoint_path or ckpt_info.checkpoint_path

    # Resolve checkpoint path relative to PiD directory
    if not os.path.isabs(ckpt_path):
        pid_dir = Path(__file__).parent.parent / "PiD"
        ckpt_path = str(pid_dir / ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"PiD checkpoint not found: {ckpt_path}\n"
            f"Please download checkpoints from https://huggingface.co/nvidia/PiD"
        )

    config_file = "pid/_src/configs/pid/config.py"

    logger.info(f"Loading PiD model: backbone={backbone}, ckpt_type={ckpt_type}")
    logger.info(f"  experiment={experiment}")
    logger.info(f"  checkpoint={ckpt_path}")

    model, _config = load_model_from_checkpoint(
        experiment_name=experiment,
        checkpoint_path=ckpt_path,
        config_file=config_file,
        enable_fsdp=False,
        experiment_opts=[],
        strict=False,
        load_ema_to_reg=False,
    )
    model.eval()

    logger.info(f"PiD model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_cached_model(backbone: str, ckpt_type: str, checkpoint_path: str | None = None) -> Any:
    """Get or load a cached PiD model."""
    cache_key = f"{backbone}:{ckpt_type}:{checkpoint_path or 'default'}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _load_pid_model(backbone, ckpt_type, checkpoint_path)
    return _MODEL_CACHE[cache_key]


def clear_model_cache():
    """Clear all cached models to free VRAM."""
    global _MODEL_CACHE
    for model in _MODEL_CACHE.values():
        del model
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("PiD model cache cleared.")


# ---------------------------------------------------------------------------
# ComfyUI ↔ PiD format conversions
# ---------------------------------------------------------------------------

def comfy_latent_to_pid(latent: dict) -> torch.Tensor:
    """Convert ComfyUI LATENT dict to PiD LQ_latent tensor.

    ComfyUI LATENT: {"samples": [B,C,H,W]}
    PiD expects: [B,C,H,W] tensor (same format, just extract from dict)
    """
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"Expected latent['samples'] to be Tensor, got {type(samples)}")
    return samples


def comfy_image_to_pid(image: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI IMAGE to PiD image format.

    ComfyUI IMAGE: [B, H, W, C] float32 [0, 1]
    PiD image: [B, 3, H, W] float32 [-1, 1]
    """
    if image.dim() != 4:
        raise ValueError(f"Expected image [B,H,W,C], got shape {image.shape}")
    # [B,H,W,C] -> [B,C,H,W]
    image = image.permute(0, 3, 1, 2)
    # [0,1] -> [-1,1]
    image = image * 2.0 - 1.0
    return image


def pid_image_to_comfy(image: torch.Tensor) -> torch.Tensor:
    """Convert PiD image output to ComfyUI IMAGE format.

    PiD output: [B, 3, H, W] float32 [-1, 1]
    ComfyUI IMAGE: [B, H, W, C] float32 [0, 1]
    """
    if image.dim() == 5:
        # [B,C,1,H,W] -> [B,C,H,W]
        image = image.squeeze(2)
    if image.dim() != 4:
        raise ValueError(f"Expected image [B,C,H,W], got shape {image.shape}")
    # [-1,1] -> [0,1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)
    # [B,C,H,W] -> [B,H,W,C]
    image = image.permute(0, 2, 3, 1)
    return image


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def pid_decode(
    model: Any,
    latent: torch.Tensor,
    prompt: str,
    cfg_scale: float = 1.0,
    num_steps: int = 4,
    seed: int = 0,
    degrade_sigma: float = 0.0,
    scale: int | None = None,
) -> torch.Tensor:
    """Run PiD decode on a latent tensor.

    Args:
        model: Loaded PiD model.
        latent: [B, C, H, W] latent tensor.
        prompt: Text prompt for conditioning.
        cfg_scale: Classifier-free guidance scale.
        num_steps: Number of denoising steps (4 for distilled checkpoints).
        seed: Random seed.
        degrade_sigma: Noise level to add to latent (0.0 = clean).
        scale: Output upscale factor. If None, uses model's default.

    Returns:
        Decoded image tensor [B, 3, H_out, W_out] in [-1, 1].
    """
    device = "cuda"
    dtype = torch.bfloat16

    latent = latent.to(device=device, dtype=dtype)
    B = latent.shape[0]

    # Derive output image size from latent shape
    latent_h, latent_w = latent.shape[-2], latent.shape[-1]
    vae_compression = getattr(model.vae_encoder, "spatial_compression_factor", 8)

    if scale is None:
        # Try to get scale from model config
        scale = getattr(model.config, "image_size", 1024) // (latent_h * vae_compression)
        if scale < 1:
            scale = 4  # Default fallback

    vae_h = latent_h * vae_compression
    vae_w = latent_w * vae_compression
    target_hw = (vae_h * scale, vae_w * scale)

    logger.info(
        f"PiD decode: latent={tuple(latent.shape)} "
        f"vae_native=({vae_h}x{vae_w}) target=({target_hw[0]}x{target_hw[1]}) "
        f"steps={num_steps} cfg={cfg_scale} sigma={degrade_sigma}"
    )

    # Optional noise injection
    if degrade_sigma > 0:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
        latent = (1.0 - degrade_sigma) * latent + degrade_sigma * noise

    # Build data batch
    data_batch = {
        model.config.input_caption_key: [prompt] * B,
        "LQ_video_or_image": torch.zeros(B, 3, vae_h, vae_w, device=device, dtype=dtype),
        "LQ_latent": latent,
        "degrade_sigma": torch.full((B,), degrade_sigma, device=device, dtype=torch.float32),
    }

    with torch.no_grad():
        output = model.generate_samples_from_batch(
            data_batch,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            image_size=target_hw,
        )

    # output is [B, 3, H_out, W_out] in [-1, 1] (or [B, 3, 1, H, W] from generate_samples_from_batch)
    return output


def pid_encode_image(model: Any, image: torch.Tensor) -> torch.Tensor:
    """Encode an image through PiD's frozen VAE.

    Args:
        model: Loaded PiD model.
        image: [B, 3, H, W] in [-1, 1].

    Returns:
        Latent [B, C, H/8, W/8].
    """
    with torch.no_grad():
        latent = model.encode_lq_latent(image)
    return latent
