# PixelDiT T2I model — inference subset.
#
# Provides the bare minimum needed by PidDistillModel: net + frozen text
# encoder + caption embedding helper + a flow-matching `timescale` field.
# Training-time machinery (EMA, REPA, flow-matching trainer, training/validation
# steps) has been removed.

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import attrs
import torch
import torch.nn as nn
from torch import Tensor

from pid_core._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from pid_core._ext.imaginaire.model import ImaginaireModel
from pid_core._ext.imaginaire.utils import misc
from pid_core._src.utils.context_parallel import broadcast as cp_broadcast
from pid_core._src.utils.context_parallel import robust_broadcast

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None  # CP is opt-in; gracefully degrade when megatron is absent

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class _EMAStubConfig:
    """Minimal stub kept so that DCP ModelWrapper.state_dict() can read `config.ema.enabled`."""

    enabled: bool = False
    rate: float = 0.1
    iteration_shift: int = 0


@attrs.define(slots=False)
class PixelDiTModelConfig:
    net: Any = None
    precision: str = "bfloat16"
    ema: _EMAStubConfig = attrs.Factory(_EMAStubConfig)

    input_data_key: str = "image"
    input_caption_key: str = "caption"

    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    model_max_length: int = 300
    chi_prompt: Any = None
    conditioner: Any = None

    # Flow matching: only `fm_timescale` is read at inference (network expects
    # t * timescale as its scalar timestep input).
    fm_timescale: float = 1000.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    prediction_type: str = "velocity"

    shift: float = 4.0
    cfg_scale: float = 2.75
    image_size: int = 1024
    negative_prompt: str = "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, watermark"
    num_sample_steps: int = 50

    dynamic_shift: dict | None = None


_TEXT_ENCODER_DICT = {
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
}


def _load_text_encoder(name: str, device: str = "cuda"):
    import os
    from pathlib import Path

    import torch.distributed as dist
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    assert name in _TEXT_ENCODER_DICT, f"Unsupported text encoder: {name}"
    model_id = _TEXT_ENCODER_DICT[name]

    is_distributed = dist.is_initialized()
    is_rank0 = (not is_distributed) or (dist.get_rank() == 0)

    if is_distributed and not is_rank0:
        dist.barrier()

    # ------------------------------------------------------------------
    # 1.  Determine local paths
    # ------------------------------------------------------------------
    # Plugin-bundled config + tokenizer (pre-downloaded into the repo).
    plugin_root = Path(__file__).parent.parent.parent.parent.resolve()
    local_config_dir = plugin_root / "pretrained" / "text_encoder" / name.replace("-", "_").replace(".", "_")
    # Also try the raw name with hyphens
    if not local_config_dir.exists():
        local_config_dir = plugin_root / "pretrained" / "text_encoder" / name

    # User-provided weights (ComfyUI standard clip folder).
    # Navigate from plugin root up to ComfyUI base, then into models/clip/.
    plugin_root = Path(__file__).parent.parent.parent.parent.resolve()
    comfy_root = plugin_root.parent.parent  # .../ComfyUI/custom_nodes/Comfyui-PiD -> .../ComfyUI
    clip_path = comfy_root / "models" / "clip" / f"{name}.safetensors"
    if not clip_path.exists():
        # Fallback: try from cwd (when ComfyUI is started from its root).
        clip_path = Path("models") / "clip" / f"{name}.safetensors"

    _local_only = os.environ.get("PID_FORCE_LOCAL_HF", "1") != "0"

    # ------------------------------------------------------------------
    # 2.  Load tokenizer
    # ------------------------------------------------------------------
    if local_config_dir.exists():
        logger.info(f"Loading tokenizer from local config dir: {local_config_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(local_config_dir), local_files_only=True)
    elif _local_only:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # 3.  Load model config
    # ------------------------------------------------------------------
    if local_config_dir.exists():
        logger.info(f"Loading model config from local config dir: {local_config_dir}")
        config = AutoConfig.from_pretrained(str(local_config_dir), local_files_only=True)
    elif _local_only:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    else:
        config = AutoConfig.from_pretrained(model_id)

    # ------------------------------------------------------------------
    # 4.  Load model weights
    # ------------------------------------------------------------------
    # PiD text encoders are large (~2.5B params).  Using from_config +
    # load_state_dict takes 2+ min because from_config materialises every
    # parameter on CPU first.  from_pretrained with low_cpu_mem_usage
    # skips that step and loads directly into the model, cutting the time
    # to <1 s.  We hard-link the safetensors into the config dir so
    # from_pretrained can find it.
    _model_link = local_config_dir / "model.safetensors"
    if clip_path.exists() and not _model_link.exists():
        try:
            os.link(str(clip_path), str(_model_link))
            logger.info(f"Hard-linked weights into config dir: {_model_link}")
        except OSError:
            import shutil

            shutil.copy2(str(clip_path), str(_model_link))
            logger.info(f"Copied weights into config dir: {_model_link}")

    if _model_link.exists() or (local_config_dir / "model.safetensors.index.json").exists():
        logger.info("Loading text encoder with low_cpu_mem_usage (fast path)...")
        full_model = AutoModelForCausalLM.from_pretrained(
            str(local_config_dir),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        logger.warning(
            f"No local safetensors found at {clip_path}. "
            f"Attempting HuggingFace cache fallback."
        )
        if _local_only:
            full_model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, low_cpu_mem_usage=True
            )
        else:
            full_model = AutoModelForCausalLM.from_pretrained(
                model_id, low_cpu_mem_usage=True
            )

    # Load onto CPU to save VRAM; _encode_text_raw will temporarily move
    # the model to GPU only while encoding the prompt.
    text_encoder = full_model.get_decoder().to(device="cpu")
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    if is_distributed and is_rank0:
        dist.barrier()

    return tokenizer, text_encoder


class _FlowMatchingTimescale(nn.Module):
    """Tiny stand-in for the deleted `FlowMatchingTrainer` — only `timescale` is read."""

    def __init__(self, timescale: float):
        super().__init__()
        self.timescale = timescale


class PixelDiTModel(ImaginaireModel):
    SUPPORTS_CONTEXT_PARALLEL: bool = False

    def __init__(self, config: PixelDiTModelConfig, text_encoder: Any = None):
        super().__init__()
        self.config = config

        # Chi-prompt: prepend a system prompt that guides Gemma to generate richer
        # embeddings for short / simple prompts.  Tokenizer pads to
        # chi_tokens + model_max_length - 2, then select_index drops the chi-prompt
        # prefix so only the enhanced user prompt (+ BOS) reaches the network.
        self._chi_prompt_str = ""
        self._num_chi_tokens = 0
        if config.chi_prompt is not None:
            self._chi_prompt_str = "\n".join(config.chi_prompt)
            logger.info("PiD chi_prompt enabled")

        if config.dynamic_shift is not None:
            _ds = config.dynamic_shift
            logger.info(
                f"PixelDiT dynamic shift: base_shift={_ds['base_shift']} "
                f"base_image_size={_ds['base_image_size_for_shift_calc']}"
            )

        _dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        requested_dtype = _dtype_map[config.precision]
        # Inference-only: bfloat16 weights + autocast so float32 intermediates
        # (RoPE, RMSNorm, etc.) are automatically cast instead of raising dtype errors.
        self.autocast_dtype = requested_dtype if requested_dtype != torch.float32 else torch.bfloat16
        self.precision = self.autocast_dtype
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        with misc.timer("PixelDiTModel: build_net"):
            self.net = lazy_instantiate(config.net)
            self.net = self.net.to(device="cuda", dtype=self.precision)
            self.net.requires_grad_(True)
            if hasattr(self.net, "init_weights"):
                self.net.init_weights()
            logger.info(f"PixDiT_T2I params: {sum(p.numel() for p in self.net.parameters()):,}")

        # Frozen text encoder. Use object.__setattr__ so DCP / nn.Module don't try to
        # register it as a child / save it in state_dict.
        # If an external text_encoder is provided (e.g. from cache), reuse it to avoid
        # reloading the 2.5B model when switching backbone / ckpt_type.
        if text_encoder is not None:
            logger.info("PixelDiTModel: reusing external text encoder (cached)")
            object.__setattr__(self, "tokenizer", text_encoder["tokenizer"])
            object.__setattr__(self, "text_encoder", text_encoder["model"])
        else:
            with misc.timer("PixelDiTModel: load_text_encoder"):
                _tokenizer, _text_encoder = _load_text_encoder(config.text_encoder_name, device="cuda")
                object.__setattr__(self, "tokenizer", _tokenizer)
                object.__setattr__(self, "text_encoder", _text_encoder)
        self._null_caption_embs = self._encode_text_raw([config.negative_prompt if config.negative_prompt else ""])[
            0
        ]

        # Tiny flow-matching shim: only `timescale` is consumed by inference.
        self.fm_trainer = _FlowMatchingTimescale(config.fm_timescale)

        self.conditioner = lazy_instantiate(config.conditioner)
        logger.info(f"PixelDiT conditioner: {self.conditioner}")

    # ---------------------------------------------------------------------
    # Text encoding
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text_raw(self, captions: list[str]) -> tuple[Tensor, Tensor]:
        if self._chi_prompt_str:
            prompts_all = [self._chi_prompt_str + cap for cap in captions]
            # Lazy count chi tokens on first call (tokenizer is now loaded).
            if self._num_chi_tokens == 0:
                self._num_chi_tokens = len(self.tokenizer(self._chi_prompt_str)["input_ids"])
                logger.info(f"PiD chi_prompt token count: {self._num_chi_tokens}")
            max_length_all = self._num_chi_tokens + self.config.model_max_length - 2
        else:
            prompts_all = captions
            max_length_all = self.config.model_max_length

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        actual_len = int(caption_token.attention_mask.sum(dim=1)[0].item())

        # Text encoder lives on CPU to save VRAM.  Move to GPU only while
        # encoding, then move back immediately.
        te_device = next(self.text_encoder.parameters()).device
        if te_device.type == "cpu":
            self.text_encoder = self.text_encoder.to("cuda")

        with torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext():
            caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0]

        if te_device.type == "cpu":
            self.text_encoder = self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # Official PiD: select_index drops the chi_prompt prefix (if present)
        # and keeps BOS + the last (model_max_length-1) positions.
        logger.info(f"_encode_text_raw: actual_len={actual_len}, max_length_all={max_length_all}")
        select_index = [0] + list(range(-self.config.model_max_length + 1, 0))
        caption_embs = caption_embs[:, select_index]
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks

    def _normalize_image(self, img: Tensor) -> Tensor:
        if img.dtype == torch.uint8:
            return img.float() / 127.5 - 1.0
        elif img.max() > 1.0:
            return img.float() / 127.5 - 1.0
        else:
            if img.min() >= 0:
                return img.float() * 2.0 - 1.0
            return img.float()

    # ---------------------------------------------------------------------
    # Context-parallel helpers (no-op when megatron CP isn't initialized).
    # ---------------------------------------------------------------------

    @staticmethod
    def get_context_parallel_group():
        if parallel_state is not None and parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def _maybe_enable_cp_on_nets(self, nets: list) -> None:
        cp_group = self.get_context_parallel_group()
        for net in nets:
            if net is None:
                continue
            if cp_group is None or cp_group.size() <= 1:
                if hasattr(net, "disable_context_parallel") and getattr(net, "is_context_parallel_enabled", False):
                    net.disable_context_parallel()
            else:
                if hasattr(net, "enable_context_parallel"):
                    net.enable_context_parallel(cp_group)

    def _broadcast_tensor_for_cp(self, t: Tensor | None) -> Tensor | None:
        cp_group = self.get_context_parallel_group()
        if t is None or cp_group is None or cp_group.size() <= 1:
            return t
        from torch.distributed import get_process_group_ranks

        src = min(get_process_group_ranks(cp_group))
        return robust_broadcast(t.contiguous(), src=src, pg=cp_group)

    def _broadcast_object_for_cp(self, obj):
        return cp_broadcast(obj, self.get_context_parallel_group())

    # ---------------------------------------------------------------------
    # Checkpoint helpers — the distill subclass overrides these for its
    # net.* / fake_score.* / discriminator.* prefix routing.
    # ---------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(prefix="net.")

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        has_core_keys = any(k.startswith("core.") for k in state_dict)
        has_net_keys = any(k.startswith("net.") for k in state_dict)

        if has_core_keys and not has_net_keys:
            logger.info("Loading original PixelDiT checkpoint (core.* prefix)")
            net_sd = {}
            for k, v in state_dict.items():
                if k == "pos_embed":
                    continue
                if k.startswith("core."):
                    net_sd[k[len("core.") :]] = v
            self.net.load_state_dict(net_sd, strict=False, assign=assign)
        else:
            _net_sd = {
                k[len("net.") :]: v
                for k, v in state_dict.items()
                if k.startswith("net.") and not k.startswith("net_ema.")
            }
            if _net_sd:
                self.net.load_state_dict(_net_sd, strict=strict, assign=assign)
