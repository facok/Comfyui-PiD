"""ComfyUI-PiD nodes — V3 API.

Nodes:
- PiDModelLoader: Load PiD checkpoint
- PiDDecode: Decode latent with PiD (replaces VAE decode)
- PiDDecodeFromImage: Image → encode → PiD decode (full pipeline)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from core.pid_model_manager import (
    comfy_image_to_pid,
    comfy_latent_to_pid,
    get_cached_model,
    pid_decode,
    pid_encode_image,
    pid_image_to_comfy,
)

logger = logging.getLogger(__name__)

# Custom type for passing loaded PiD models between nodes
PID_MODEL = io.Custom("PID_MODEL")

_BACKBONE_OPTIONS = ["flux", "flux2", "sd3", "zimage", "rae", "scale_rae"]
_CKPT_TYPE_OPTIONS = ["2k", "2kto4k"]


# =============================================================================
# PiD Model Loader
# =============================================================================

class PiDModelLoader(io.ComfyNode):
    """Load a PiD (Pixel Diffusion Decoder) model checkpoint."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PiDModelLoader",
            display_name="PiD Model Loader",
            category="PiD",
            description="Load a PiD checkpoint. Checkpoints should be downloaded from https://huggingface.co/nvidia/PiD",
            inputs=[
                io.Combo.Input(
                    "backbone",
                    options=_BACKBONE_OPTIONS,
                    default="flux",
                    tooltip="VAE backbone that the PiD model was trained with",
                ),
                io.Combo.Input(
                    "ckpt_type",
                    options=_CKPT_TYPE_OPTIONS,
                    default="2k",
                    tooltip="'2k' = 2048px decoder (4x upscaling). '2kto4k' = up to 4K (flux/flux2/sd3/zimage only)",
                ),
                io.String.Input(
                    "checkpoint_path",
                    default="",
                    tooltip="Optional: explicit checkpoint path. If empty, uses the official registry default.",
                ),
            ],
            outputs=[
                PID_MODEL.Output("PID_MODEL", display_name="PiD Model"),
            ],
        )

    @classmethod
    def execute(cls, backbone: str, ckpt_type: str, checkpoint_path: str):
        ckpt_path = checkpoint_path.strip() or None
        model = get_cached_model(backbone, ckpt_type, ckpt_path)
        # Return a lightweight handle; actual model stays in global cache
        return io.NodeOutput({
            "model": model,
            "backbone": backbone,
            "ckpt_type": ckpt_type,
        })


# =============================================================================
# PiD Decode — replace VAE decode with PiD super-resolution
# =============================================================================

class PiDDecode(io.ComfyNode):
    """Decode a latent using PiD for high-resolution super-resolution output.

    Replaces the standard VAE Decode step in ComfyUI workflows.
    Takes a ComfyUI LATENT (from KSampler) and produces a high-res IMAGE.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PiDDecode",
            display_name="PiD Decode",
            category="PiD",
            description="Decode latent with PiD pixel diffusion decoder. Output resolution = latent * 8 * scale.",
            inputs=[
                io.Latent.Input("latent", tooltip="ComfyUI LATENT from KSampler"),
                PID_MODEL.Input("pid_model", tooltip="Loaded PiD model from PiD Model Loader"),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt describing the desired image content",
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=1.0,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Classifier-free guidance scale. 1.0 = no CFG",
                ),
                io.Int.Input(
                    "num_steps",
                    default=4,
                    min=1,
                    max=50,
                    step=1,
                    tooltip="PiD denoising steps. Use 4 for official distilled checkpoints.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Random seed for PiD decoder",
                ),
                io.Float.Input(
                    "degrade_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Optional noise to add to latent before decode (0 = clean). Useful for stochastic decoding.",
                ),
                io.Int.Input(
                    "scale",
                    default=0,
                    min=0,
                    max=8,
                    step=1,
                    tooltip="Upscale factor. 0 = auto (from model config, typically 4x).",
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE", tooltip="Super-resolved output image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: dict,
        pid_model: dict,
        prompt: str,
        cfg_scale: float,
        num_steps: int,
        seed: int,
        degrade_sigma: float,
        scale: int,
    ):
        model = pid_model["model"]
        latent_tensor = comfy_latent_to_pid(latent)

        # Ensure prompt is not empty
        if not prompt or not prompt.strip():
            prompt = "high quality image"

        actual_scale = scale if scale > 0 else None

        output = pid_decode(
            model=model,
            latent=latent_tensor,
            prompt=prompt.strip(),
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            scale=actual_scale,
        )

        comfy_image = pid_image_to_comfy(output)
        return io.NodeOutput(comfy_image)


# =============================================================================
# PiD Decode From Image — full encode + decode pipeline
# =============================================================================

class PiDDecodeFromImage(io.ComfyNode):
    """Image → VAE encode → optional noise → PiD decode.

    Useful for upscaling existing images with PiD's super-resolution decoder.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PiDDecodeFromImage",
            display_name="PiD Decode From Image",
            category="PiD",
            description="Encode image to latent, optionally add noise, then decode with PiD for SR.",
            inputs=[
                io.Image.Input("image", tooltip="Input image to super-resolve"),
                PID_MODEL.Input("pid_model", tooltip="Loaded PiD model"),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt describing the image",
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=1.0,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                ),
                io.Int.Input(
                    "num_steps",
                    default=4,
                    min=1,
                    max=50,
                    step=1,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "degrade_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Noise level to add to encoded latent (0 = clean round-trip)",
                ),
                io.Int.Input(
                    "scale",
                    default=0,
                    min=0,
                    max=8,
                    step=1,
                    tooltip="Upscale factor. 0 = auto.",
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE", tooltip="Super-resolved output"),
                io.Latent.Output("LATENT", tooltip="Encoded latent (optional debug output)"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        pid_model: dict,
        prompt: str,
        cfg_scale: float,
        num_steps: int,
        seed: int,
        degrade_sigma: float,
        scale: int,
    ):
        model = pid_model["model"]
        pid_image = comfy_image_to_pid(image)

        # Encode through PiD's frozen VAE
        latent = pid_encode_image(model, pid_image)

        if not prompt or not prompt.strip():
            prompt = "high quality image"

        actual_scale = scale if scale > 0 else None

        # Decode with PiD
        output = pid_decode(
            model=model,
            latent=latent,
            prompt=prompt.strip(),
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            scale=actual_scale,
        )

        comfy_image = pid_image_to_comfy(output)

        # Build ComfyUI LATENT dict for optional passthrough
        latent_dict = {"samples": latent.cpu().float()}

        return io.NodeOutput(comfy_image, latent_dict)


# =============================================================================
# Extension registration
# =============================================================================

class PiDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PiDModelLoader, PiDDecode, PiDDecodeFromImage]


async def comfy_entrypoint() -> PiDExtension:
    return PiDExtension()
