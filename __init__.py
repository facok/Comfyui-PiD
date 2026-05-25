"""ComfyUI-PiD — Pixel Diffusion Decoder plugin for ComfyUI."""

import os

# Register ComfyUI/models/PiD as a standard model folder before anything else
import folder_paths

_pid_dir = os.path.join(folder_paths.base_path, "models", "PiD")
os.makedirs(_pid_dir, exist_ok=True)
folder_paths.add_model_folder_path("pid", _pid_dir)

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
