"""ComfyUI-PiD nodes — V3 API.

Nodes:
- PiDModelLoader: Load PiD checkpoint
- PiDDecode: Decode latent with PiD (replaces VAE decode)
"""

from __future__ import annotations

import logging
from typing import Any

from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, io

from .core.pid_model_manager import (
    comfy_latent_to_pid,
    get_cached_model,
    run_pid_decode,
)

logger = logging.getLogger(__name__)

PID_MODEL = io.Custom("PID_MODEL")

_BACKBONE_OPTIONS = ["flux", "flux2", "sd3", "zimage", "rae", "scale_rae"]
_CKPT_TYPE_OPTIONS = ["2k", "2kto4k"]


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
                io.Combo.Input(
                    "checkpoint_name",
                    options=folder_paths.get_filename_list("pid") or ["(none - download from huggingface.co/nvidia/PiD)"],
                    default="",
                    tooltip="Select a PiD checkpoint from ComfyUI/models/PiD/. Leave empty to use the official registry default.",
                ),
            ],
            outputs=[
                PID_MODEL.Output("PID_MODEL", display_name="PiD Model"),
            ],
        )

    @classmethod
    def execute(cls, backbone: str, ckpt_type: str, checkpoint_name: str):
        ckpt_path = None
        if checkpoint_name and not checkpoint_name.startswith("("):
            ckpt_path = folder_paths.get_full_path("pid", checkpoint_name)
        model = get_cached_model(backbone, ckpt_type, ckpt_path)
        return io.NodeOutput({
            "model": model,
            "backbone": backbone,
            "ckpt_type": ckpt_type,
        })


class PiDDecode(io.ComfyNode):
    """Decode a latent using PiD for high-resolution super-resolution output."""

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
                    tooltip="Upscale factor. 0 = auto (from model config, always 4x for flux/zimage/sd3/flux2/rae, 8x for scale_rae). Non-matching values are ignored because PiD checkpoints have a fixed SR ratio baked into the network.",
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
        num_steps: int,
        seed: int,
        degrade_sigma: float,
        scale: int,
    ):
        return io.NodeOutput(run_pid_decode(
            model=pid_model["model"],
            latent=comfy_latent_to_pid(latent),
            prompt=prompt,
            cfg_scale=1.0,  # distilled checkpoints do not use CFG
            num_steps=num_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            scale=scale,
            backbone=pid_model["backbone"],
        ))

class PiDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PiDModelLoader, PiDDecode]


async def comfy_entrypoint() -> PiDExtension:
    return PiDExtension()
