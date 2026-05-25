"""Download PiD text encoder config & tokenizer files for offline use.

Run this on a machine with internet access:
    python scripts/download_text_encoder.py

This downloads only the small config/tokenizer files (~5-10MB), NOT the
large model weights. The weights (safetensors) are expected to be provided
by the user separately in ComfyUI/models/clip/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def download(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None):
    """Download files from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir_use_symlinks=False,
    )
    print(f"Done. Files saved to: {local_dir}")


def main():
    plugin_root = Path(__file__).parent.parent.resolve()
    pretrained_dir = plugin_root / "pretrained" / "text_encoder"

    # gemma-2-2b-it (default text encoder for PiD flux/flux2/sd3)
    gemma_dir = pretrained_dir / "gemma-2-2b-it"
    print("=" * 60)
    print("Downloading gemma-2-2b-it config & tokenizer files...")
    print("(Model weights are excluded - provide your own safetensors)")
    print("=" * 60)
    download(
        repo_id="Efficient-Large-Model/gemma-2-2b-it",
        local_dir=gemma_dir,
        ignore_patterns=["*.safetensors", "*.bin", "*.msgpack", "*.h5", "*.pt", "*.pth"],
    )

    print()
    print("=" * 60)
    print("All done!")
    print(f"Config/tokenizer files are in: {pretrained_dir}")
    print()
    print("Next steps:")
    print("  1. Copy the ENTIRE pretrained/ directory to your offline machine")
    print("  2. Place your gemma-2-2b-it.safetensors in:")
    print("     ComfyUI/models/clip/gemma-2-2b-it.safetensors")
    print("  3. PiD will load from local files automatically")
    print("=" * 60)


if __name__ == "__main__":
    main()
