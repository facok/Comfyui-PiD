"""ComfyUI-PiD — Pixel Diffusion Decoder plugin for ComfyUI.

PiD is a plug-and-play diffusion decoder that replaces VAE/RAE decoders,
turning latent representations directly into super-resolved pixels.

Official repo: https://github.com/nvidia/PiD
Paper: https://arxiv.org/abs/2605.23902
Model weights: https://huggingface.co/nvidia/PiD

Nodes:
- PiDModelLoader: Load PiD checkpoints
- PiDDecode: Decode ComfyUI LATENT with PiD for super-resolution
- PiDDecodeFromImage: Encode image + PiD decode for upscaling
"""

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
