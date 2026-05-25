"""ComfyUI-PiD — Pixel Diffusion Decoder plugin for ComfyUI."""

import os

import folder_paths

# Must register before importing nodes so folder_paths.get_filename_list("pid")
# works inside PiDModelLoader.define_schema().
_pid_dir = os.path.join(folder_paths.base_path, "models", "PiD")
os.makedirs(_pid_dir, exist_ok=True)
folder_paths.add_model_folder_path("pid", _pid_dir)

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
