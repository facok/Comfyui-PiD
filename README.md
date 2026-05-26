# ComfyUI-PiD — Pixel Diffusion Decoder

> **Unofficial / 非官方实现** — This is an experimental community wrapper. For the official PiD repository, see [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD). It is recommended to wait for an official ComfyUI implementation from the PiD authors or ComfyUI maintainers for production use.
>
> 这是一个实验性的社区封装插件。官方 PiD 仓库见 [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD)。建议等待 PiD 作者或 ComfyUI 维护者提供官方实现后再用于生产环境。

PiD (Pixel Diffusion Decoder) is a plug-and-play diffusion-based decoder that replaces the traditional VAE decoder, directly turning latent representations into high-resolution, super-resolved pixels. This is the **ComfyUI V3 API** wrapper for NVIDIA PiD.

PiD（Pixel Diffusion Decoder）是一个即插即用的扩散解码器，替代传统 VAE 解码器，将 latent 直接转换为高分辨率像素。这是 NVIDIA PiD 的 **ComfyUI V3 API** 封装插件。

---

## Installation / 安装

Clone into your `ComfyUI/custom_nodes/` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/Comfyui-PiD.git
```

Dependencies will be installed automatically on first launch.

依赖会在首次启动时自动安装。

If automatic installation fails, install manually with ComfyUI's embedded Python:

如果自动安装失败，使用 ComfyUI 自带的 Python 手动安装：

```bash
# Windows portable
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Comfyui-PiD\requirements.txt

# Linux / MacOS
python -m pip install -r ComfyUI/custom_nodes/Comfyui-PiD/requirements.txt
```

---

## Model Setup / 模型准备

### 1. PiD Checkpoints / PiD 权重

Download PiD checkpoints from [Hugging Face — nvidia/PiD](https://huggingface.co/nvidia/PiD).

Place them in `ComfyUI/models/PiD/` (flat structure, **no** `checkpoints/` subfolder):

从 [Hugging Face — nvidia/PiD](https://huggingface.co/nvidia/PiD) 下载 PiD 权重，放到 `ComfyUI/models/PiD/`（扁平结构，**不要** `checkpoints/` 子目录）：

```
ComfyUI/models/PiD/
  PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth
  PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth
  ...
```

### 2. Text Encoder / 文本编码器

PiD uses **gemma-2-2b-it** as its text encoder (~2.5B params).

PiD 使用 **gemma-2-2b-it** 作为文本编码器（约 25 亿参数）。

- Download the **single `.safetensors` weight file** and place it at:  
  下载**单个 `.safetensors` 权重文件**，放到：
  ```
  ComfyUI/models/clip/gemma-2-2b-it.safetensors
  ```
  Direct link / 直接下载：
  ```
  https://huggingface.co/Efficient-Large-Model/gemma-2-2b-it/resolve/main/gemma-2-2b-it.safetensors?download=true
  ```
- The tokenizer & config files are **bundled** with this plugin under `pretrained/text_encoder/gemma-2-2b-it/`. No HuggingFace download is needed at runtime.  
  tokenizer 和配置文件已**预打包**在 `pretrained/text_encoder/gemma-2-2b-it/` 目录下，运行时不需要联网下载。

---

## Nodes / 节点

### PiD Model Loader

Loads a PiD checkpoint.

| Parameter | Description | 说明 |
|-----------|-------------|------|
| **backbone** | VAE backbone: `flux`, `zimage` | PiD 训练时用的 VAE backbone |
| **ckpt_type** | `2k` = 512×512 input → 2048×2048 output. `2kto4k` = 1024×1024 input → up to 4096×4096 output. | `2k` 输入限制 **512×512**，`2kto4k` 输入限制 **1024×1024** |
| **checkpoint_name** | Optional: select a custom `.pth/.safetensors` from `ComfyUI/models/PiD/`. Leave empty to use the official registry default. | 可选：从 `ComfyUI/models/PiD/` 选择自定义权重，留空使用官方默认 |

> **Important / 重要 — Input Size Limits / 输入尺寸限制**：
> - **`2k`** — Input must be **512×512** (latent 64×64). Do NOT use with 1024×1024 inputs — causes blur and color shift. Output = 2048×2048.
>   `2k` 输入必须是 **512×512**（latent 64×64），**不可用于 1024×1024** — 会导致模糊和变色。输出 2048×2048。
> - **`2kto4k`** — Input must be **1024×1024** (latent 128×128). Do NOT use with 512×512 inputs. Output = 4096×4096.
>   `2kto4k` 输入必须是 **1024×1024**（latent 128×128），**不可用于 512×512**。输出 4096×4096。

### PiD Decode

Decodes a ComfyUI latent through PiD for super-resolution output.

| Parameter | Default | Description | 说明 |
|-----------|---------|-------------|------|
| **latent** | — | ComfyUI `LATENT` output from KSampler | 来自 KSampler 的 ComfyUI `LATENT` |
| **pid_model** | — | Loaded PiD model from PiD Model Loader | PiD Model Loader 加载的模型 |
| **prompt** | `""` | Text prompt describing the image content. Falls back to `"high quality image"` if empty. | 描述图像内容的文本提示，为空则回退到 `"high quality image"` |
| **num_steps** | `4` | PiD denoising steps. **Use 4** for official distilled checkpoints. | PiD 去噪步数，官方蒸馏模型请用 **4** |
| **seed** | `0` | Random seed for reproducibility | 随机种子 |
| **degrade_sigma** | `0.0` | Optional noise to add to the latent before decode. `0` = clean deterministic decode. | 解码前加到 latent 的噪声，`0` 表示确定性的干净解码 |

---

## Recommended Workflow / 推荐工作流

```
Load Checkpoint (Z-Image / FLUX / SD3)
  ↓
CLIP Text Encode (prompt)
  ↓
KSampler / Sampler → LATENT (1024×1024 or 512×512)
  ↓
PiD Model Loader (backbone=zimage, ckpt_type=2kto4k)
  ↓
PiD Decode (num_steps=4, prompt=same as above)
  ↓
Save Image / Preview Image
```

**Example / 示例**：
- Z-Image-Turbo KSampler → 1024×1024 latent  
- PiD Model Loader: `backbone=zimage`, `ckpt_type=2kto4k`  
- PiD Decode: fixed 4× upscale → 4096×4096, `num_steps=4`

---

## Notes / 注意事项

1. **No VAE loading inside PiD / PiD 内部不加载 VAE**  
   PiD uses ComfyUI's native VAE encode/decode. The model only loads the diffusion decoder and text encoder.  
   PiD 使用 ComfyUI 原生的 VAE 编解码，模型内部只加载扩散解码器和文本编码器。

2. **bfloat16 by default / 默认 bfloat16**  
   Both the PiD decoder and text encoder run in `bfloat16` for VRAM efficiency. The text encoder is kept on CPU and moved to GPU only during prompt encoding.  
   PiD 解码器和文本编码器均以 `bfloat16` 运行以节省显存。文本编码器常驻 CPU，仅在编码 prompt 时临时上 GPU。

3. **One model cached at a time / 一次只缓存一个模型**  
   To save VRAM, only one PiD model is kept in memory. Loading a different backbone/ckpt will clear the previous one.  
   为节省显存，一次只缓存一个 PiD 模型。加载不同 backbone/ckpt 会自动释放上一个。

4. **VRAM estimate / 显存估算**  
   - 1024→4096 (scale=4): ~14-18GB VRAM  
   - 512→2048 (scale=4): ~8-12GB VRAM  
   If OOM, try a smaller input latent or reduce batch size.  
   如果显存不足，请缩小输入 latent 尺寸或减少 batch size。

5. **Height limit / 高度限制**
   The `2kto4k` model was trained up to ~4096px output height. Beyond this, RoPE position encoding extrapolation may cause **green artifacts** at the bottom of the image.
   `2kto4k` 模型训练时最大输出高度约为 4096px，超过此高度时 RoPE 位置编码外推可能导致图像**底部出现绿色伪影**。

   **Pixel ↔ Latent conversion / 像素与 latent 换算**：
   - flux VAE compression = 8, PiD sr4x scale = 4
   - 输出像素高度 = `latent_height × 8 × 4` = `latent_height × 32`
   - 示例：latent 128×128 → 输出 4096×4096

   > To stay within the safe range, keep **latent height ≤ 128** (output ≤ 4096px). 安全范围：**latent 高度 ≤ 128**（输出 ≤ 4096px）。

---

## Troubleshooting / 故障排查

| Issue | Cause | Fix |
|-------|-------|-----|
| Blurry output + color shift / 模糊+变色 | **2k** used with 1024×1024 input (exceeds 512 limit) / 2k 输入超过 512 限制 | Switch to `2kto4k`, ensure input is 1024×1024 |
| Blurry output + color shift / 模糊+变色 | **2kto4k** used with 512×512 input (below 1024) / 2kto4k 输入低于 1024 | Switch to `2k`, ensure input is 512×512 |
| Green artifacts at bottom / 底部绿色伪影 | Output height > 4096px (beyond 2kto4k training distribution) | Reduce input latent height to ≤128 |
| OOM at 4096px / 4096px 显存溢出 | Target resolution too large | Reduce input latent size (e.g. 768×768 → 1024×1024 still works with 2kto4k) |
| PiD checkpoint not found / 找不到权重 | Model not in `ComfyUI/models/PiD/` | Download from HF and place correctly |

---

## Credits / 致谢

- **NVIDIA PiD** (official): [Hugging Face — nvidia/PiD](https://huggingface.co/nvidia/PiD) | [GitHub — nv-tlabs/PiD](https://github.com/nv-tlabs/PiD)
- Paper: arXiv:2605.23902
