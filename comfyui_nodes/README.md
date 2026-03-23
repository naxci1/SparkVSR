# SparkVSR ComfyUI Custom Nodes

SeedVR2.5-style ComfyUI interface for SparkVSR Video Super-Resolution.

## Nodes

| Node | Description |
|------|-------------|
| **SparkVSR Load Pipeline** | Load the SparkVSR model (transformer + VAE + text encoder) |
| **SparkVSR Configure VAE** | Set VAE tiling/slicing for VRAM-efficient processing |
| **SparkVSR Video Upscaler** | Run video super-resolution on a batch of ComfyUI frames |

---

## Installation

### Prerequisites

- ComfyUI installed and working
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.x (RTX 40xx / 50xx recommended)

### Step 1 — Copy nodes into ComfyUI

```bash
# Option A: copy the folder
cp -r /path/to/SparkVSR/comfyui_nodes  /path/to/ComfyUI/custom_nodes/SparkVSR

# Option B: symlink (Linux/macOS)
ln -s /path/to/SparkVSR/comfyui_nodes  /path/to/ComfyUI/custom_nodes/SparkVSR

# Option B: symlink (Windows PowerShell — run as Administrator)
New-Item -ItemType Junction `
    -Path "C:\ComfyUI\custom_nodes\SparkVSR" `
    -Target "C:\path\to\SparkVSR\comfyui_nodes"
```

### Step 2 — Install Python dependencies into ComfyUI's environment

```bash
# Activate ComfyUI's venv first, then:
pip install diffusers>=0.30.0 transformers>=4.40.0 accelerate safetensors
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pillow numpy tqdm einops sentencepiece tokenizers
```

### Step 3 — Download SparkVSR models

```python
from huggingface_hub import snapshot_download
snapshot_download("zai-org/CogVideoX1.5-5B-I2V",
                  local_dir="pretrained_weights/CogVideoX1.5-5B-I2V")
snapshot_download("JiongzeYu/SparkVSR",
                  local_dir="checkpoints/sparkvsr-s2/ckpt-500-sft")
```

### Step 4 — Restart ComfyUI

---

## Basic Workflow

```
[Load Video / Image Sequence]
         │
         ▼
[SparkVSR Load Pipeline]  ──→  [SparkVSR Configure VAE]
                                         │
                                         ▼
                            [SparkVSR Video Upscaler]  ←── image frames
                                         │
                                         ▼
                               [Save Video / Images]
```

---

## 16 GB VRAM Settings (RTX 5070 Ti / 4080)

In **SparkVSR Load Pipeline**:
- `dtype`: `bfloat16`
- `offload_device`: `cpu`

In **SparkVSR Configure VAE**:
- `decode_tiled`: ✅ True
- `decode_tile_size`: 736
- `enable_slicing`: ✅ True

In **SparkVSR Video Upscaler**:
- `batch_size`: 49  (fewer frames per chunk)
- `tile_size_h`: 480
- `tile_size_w`: 854

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: diffusers` | Run `pip install diffusers transformers accelerate` inside ComfyUI's venv |
| Out of VRAM | Enable `offload_device=cpu` + `decode_tiled=True` + lower `batch_size` |
| Slow on first run | Model is being JIT-compiled; subsequent runs are faster |
| Windows path errors | Use forward slashes `/` or raw strings `r"C:\..."` in model_path |
