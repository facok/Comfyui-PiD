"""PiD model manager — loads checkpoints and runs inference.

Handles ComfyUI ↔ PiD data format conversions.
All PiD imports are deferred to runtime to avoid crashing ComfyUI startup
when PiD dependencies are not yet installed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_TEXT_ENCODER_CACHE: dict[str, Any] = {}
_COMPAT_PATCHED = False


def _ensure_pid_in_path():
    """Ensure plugin root is in sys.path so 'pid_core' package can be imported."""
    plugin_root = Path(__file__).parent.parent.resolve()
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))


def _setup_compat():
    """Apply Windows compatibility patches (idempotent)."""
    global _COMPAT_PATCHED
    if _COMPAT_PATCHED:
        return
    from .compat import setup_pid_compat

    setup_pid_compat()
    _COMPAT_PATCHED = True


def _resolve_ckpt_path(ckpt_path: str) -> str | None:
    """Resolve checkpoint path using ComfyUI model folders.

    Registry paths include a legacy 'checkpoints/' prefix; we strip it
    so models live directly under ComfyUI/models/PiD/.
    Also tries common suffixes (.pth, .safetensors) if the exact path is missing.
    """
    if not ckpt_path:
        return None
    if os.path.isabs(ckpt_path):
        return ckpt_path

    # Strip legacy 'checkpoints/' prefix
    if ckpt_path.startswith("checkpoints/"):
        ckpt_path = ckpt_path[len("checkpoints/"):]

    import folder_paths
    for folder in folder_paths.get_folder_paths("pid"):
        candidate = os.path.join(folder, ckpt_path)
        if os.path.isfile(candidate):
            return candidate
        # Try common alternative filenames (single-file downloads from HF)
        base, _ = os.path.splitext(candidate)
        for ext in (".pth", ".pt", ".safetensors", ".bin"):
            alt = base + ext
            if os.path.isfile(alt):
                return alt
            # Also try without the /model_ema_bf16 suffix
            dir_alt = os.path.dirname(candidate)
            name = os.path.basename(dir_alt) + ext
            alt2 = os.path.join(os.path.dirname(dir_alt), name)
            if os.path.isfile(alt2):
                return alt2

    return None


def get_cached_text_encoder(name: str = "gemma-2-2b-it") -> dict[str, Any] | None:
    """Get or load a cached text encoder. All PiD checkpoints use the same
    gemma-2-2b-it encoder, so caching it avoids reloading ~2.5B params when
    switching backbone / ckpt_type.
    """
    if name not in _TEXT_ENCODER_CACHE:
        from pid_core._src.models.pixeldit_model import _load_text_encoder

        logger.info(f"Loading text encoder '{name}' (not cached)")
        tokenizer, model = _load_text_encoder(name, device="cuda")
        _TEXT_ENCODER_CACHE[name] = {"tokenizer": tokenizer, "model": model}
        logger.info(f"Text encoder '{name}' cached.")
    else:
        logger.info(f"Text encoder '{name}' loaded from cache.")
    return _TEXT_ENCODER_CACHE[name]


def clear_text_encoder_cache():
    """Clear the text encoder cache to free RAM."""
    _TEXT_ENCODER_CACHE.clear()
    logger.info("Text encoder cache cleared.")


def _load_pid_model(
    backbone: str,
    ckpt_type: str,
    checkpoint_path: str | None = None,
) -> Any:
    _ensure_pid_in_path()
    _setup_compat()

    from pid_core._src.inference.checkpoint_registry import get_pid_checkpoint
    from pid_core._src.utils.model_loader import load_model_from_checkpoint

    ckpt_info = get_pid_checkpoint(backbone, ckpt_type)
    experiment = ckpt_info.experiment

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = _resolve_ckpt_path(ckpt_info.checkpoint_path)

    if not ckpt_path or not os.path.isfile(ckpt_path):
        display_path = ckpt_info.checkpoint_path
        if display_path.startswith("checkpoints/"):
            display_path = display_path[len("checkpoints/"):]
        raise FileNotFoundError(
            f"PiD checkpoint not found for {backbone}/{ckpt_type}\n"
            f"Expected: ComfyUI/models/PiD/{display_path}\n"
            f"          or ComfyUI/models/PiD/<name>.pth\n"
            f"Download from: https://huggingface.co/nvidia/PiD"
        )

    # Disable PiD's built-in VAE loading.  ComfyUI already provides VAE
    # encode/decode nodes; PiD only needs the diffusion decoder.
    experiment_opts: list[str] = ["model.config.tokenizer=null"]

    logger.info(f"Loading PiD model: backbone={backbone}, ckpt_type={ckpt_type}")
    logger.info(f"  experiment={experiment}")
    logger.info(f"  checkpoint={ckpt_path}")

    # Reuse cached text encoder so switching backbone/ckpt doesn't reload Gemma.
    text_encoder = get_cached_text_encoder("gemma-2-2b-it")

    plugin_root = Path(__file__).parent.parent.resolve()
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    model, _ = load_model_from_checkpoint(
        experiment_name=experiment,
        checkpoint_path=ckpt_path,
        config_file="pid_core._src.configs.pid.config",
        enable_fsdp=False,
        experiment_opts=experiment_opts,
        strict=False,
        load_ema_to_reg=False,
        text_encoder=text_encoder,
    )
    model.eval()

    # Record the official pid_scale (e.g. 4 for sr4x, 8 for scale_rae) on the
    # model so _resolve_scale can auto-infer correctly.
    if not hasattr(model.config, "pid_scale"):
        object.__setattr__(model.config, "pid_scale", ckpt_info.pid_scale)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"PiD model loaded. Parameters: {param_count:,}")
    return model


def get_cached_model(backbone: str, ckpt_type: str, checkpoint_path: str | None = None) -> Any:
    """Get or load a cached PiD model. Only one model is kept at a time to save VRAM."""
    cache_key = f"{backbone}:{ckpt_type}:{checkpoint_path or 'default'}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _MODEL_CACHE[cache_key] = _load_pid_model(backbone, ckpt_type, checkpoint_path)
    return _MODEL_CACHE[cache_key]


def clear_model_cache():
    """Clear all cached models and text encoder to free VRAM/RAM."""
    _MODEL_CACHE.clear()
    _TEXT_ENCODER_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("PiD model cache cleared.")


def comfy_latent_to_pid(latent: dict) -> torch.Tensor:
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"Expected latent['samples'] to be Tensor, got {type(samples)}")
    return samples


# Mapping from our backbone names to ComfyUI's latent-format classes.
# process_in() applies the VAE-specific normalization (z = scale * (raw - shift))
# so PiD receives latents in the same space it was trained on.
_BACKBONE_LATENT_FORMAT: dict[str, str] = {
    "flux": "Flux",
    "zimage": "Flux",  # ZImage shares Flux1's 16-ch VAE
}


def normalize_comfy_latent_for_pid(latent: torch.Tensor, backbone: str) -> torch.Tensor:
    fmt_name = _BACKBONE_LATENT_FORMAT.get(backbone)
    if fmt_name is None:
        return latent
    try:
        import comfy.latent_formats as lf

        fmt_cls = getattr(lf, fmt_name)
        return fmt_cls().process_in(latent)
    except Exception:
        return latent


def pid_image_to_comfy(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 5:
        image = image.squeeze(2)
    if image.dim() != 4:
        raise ValueError(f"Expected image [B,C,H,W], got shape {image.shape}")
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0).permute(0, 2, 3, 1).float()


def pid_latent_to_comfy(latent: torch.Tensor) -> dict:
    return {"samples": latent.cpu().float()}


def sanitize_prompt(prompt: str, fallback: str = "high quality image") -> str:
    return prompt.strip() if prompt and prompt.strip() else fallback


def _get_model_device_dtype(model: Any) -> tuple[str, torch.dtype]:
    """Return (device, dtype) from the model's own parameters."""
    param = next(model.parameters())
    return param.device.type, param.dtype


def _resolve_scale(model: Any, latent_h: int, latent_w: int, vae_compression: int) -> int:
    """Return the fixed SR ratio baked into this PiD checkpoint (sr4x → 4)."""
    pid_scale = getattr(model.config, "pid_scale", None)
    if pid_scale is not None:
        return pid_scale
    # Fallback (should never hit for official checkpoints).
    image_size = getattr(model.config, "image_size", 1024)
    inferred = image_size // (latent_h * vae_compression)
    if inferred < 1:
        logger.warning(
            f"Inferred scale {inferred} from image_size={image_size}, "
            f"latent={latent_h}x{latent_w}, compression={vae_compression}. Falling back to 4x."
        )
        return 4
    return inferred


def run_pid_decode(
    model: Any,
    latent: torch.Tensor,
    prompt: str,
    cfg_scale: float,
    num_steps: int,
    seed: int,
    degrade_sigma: float,
    backbone: str,
) -> torch.Tensor:
    """Shared decode pipeline: validate prompt, run PiD decode, convert output."""
    prompt = sanitize_prompt(prompt)
    device, dtype = _get_model_device_dtype(model)

    latent = normalize_comfy_latent_for_pid(latent, backbone)
    if latent.device.type != device or latent.dtype != dtype:
        latent = latent.to(device=device, dtype=dtype)
    logger.info(
        f"PiD LQ_latent stats (after normalize): "
        f"mean={latent.float().mean().item():.4f} "
        f"std={latent.float().std().item():.4f} "
        f"min={latent.float().min().item():.4f} "
        f"max={latent.float().max().item():.4f}"
    )
    B = latent.shape[0]

    latent_h, latent_w = latent.shape[-2], latent.shape[-1]
    vae_compression = getattr(model.vae_encoder, "spatial_compression_factor", 8)
    scale = _resolve_scale(model, latent_h, latent_w, vae_compression)

    vae_h = latent_h * vae_compression
    vae_w = latent_w * vae_compression
    target_hw = (vae_h * scale, vae_w * scale)

    logger.info(
        f"PiD decode: latent={tuple(latent.shape)} "
        f"vae_native=({vae_h}x{vae_w}) target=({target_hw[0]}x{target_hw[1]}) "
        f"steps={num_steps} cfg={cfg_scale} sigma={degrade_sigma}"
    )

    if degrade_sigma > 0:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
        latent = (1.0 - degrade_sigma) * latent + degrade_sigma * noise

    lq_type = getattr(model.config, "lq_condition_type", "latent")
    data_batch: dict[str, Any] = {
        model.config.input_caption_key: [prompt] * B,
        "LQ_latent": latent,
        "degrade_sigma": torch.full((B,), degrade_sigma, device=device, dtype=dtype),
    }
    if lq_type in ("image", "image_latent"):
        data_batch["LQ_video_or_image"] = torch.zeros(B, 3, vae_h, vae_w, device=device, dtype=dtype)

    with torch.no_grad():
        output = model.generate_samples_from_batch(
            data_batch,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            image_size=target_hw,
        )

    return pid_image_to_comfy(output)

