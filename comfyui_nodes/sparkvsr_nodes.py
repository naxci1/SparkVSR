"""
SparkVSR ComfyUI nodes — SeedVR2.5-style interface.

Three nodes mirror the SeedVR2.5 panel layout:
  1. SparkVSR_LoadPipeline  ← "Load DiT Model"   (loads the CogVideoX transformer)
  2. SparkVSR_LoadVAE       ← "Load VAE Model"   (configures the VAE component)
  3. SparkVSR_VideoUpscaler ← "Video Upscaler"   (runs the actual SR)

ComfyUI image format:  BHWC float32 [0,1]
SparkVSR video format: BCFHW float32 [-1,1]   (where F = number of frames)
"""

from __future__ import annotations

import gc
import os
import sys
import struct
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("SparkVSR")

# ---------------------------------------------------------------------------
# Optional imports (graceful fallback if not installed)
# ---------------------------------------------------------------------------

try:
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
    _DIFFUSERS_OK = True
except ImportError:
    _DIFFUSERS_OK = False
    logger.warning("diffusers not found. SparkVSR nodes will be disabled.")

try:
    from safetensors.torch import load_file as load_safetensors
    _SAFETENSORS_OK = True
except ImportError:
    _SAFETENSORS_OK = False

try:
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    _ROPE_OK = True
except ImportError:
    _ROPE_OK = False

# Add SparkVSR repo root to sys.path so finetune.utils can be imported
_SPARKVSR_ROOT = Path(__file__).resolve().parent.parent
if str(_SPARKVSR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARKVSR_ROOT))

# ---------------------------------------------------------------------------
# In-memory pipeline cache  {cache_key: pipeline_object}
# ---------------------------------------------------------------------------
_PIPELINE_CACHE: Dict[str, Any] = {}
_VAE_CACHE: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Helper: dtype string → torch.dtype
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def _str_to_dtype(s: str) -> torch.dtype:
    return _DTYPE_MAP.get(s.lower(), torch.bfloat16)


# ---------------------------------------------------------------------------
# Helper: free GPU memory
# ---------------------------------------------------------------------------

def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Helper: comfyui IMAGE (BHWC) ↔ video tensor (BCFHW / BFCHW)
# ---------------------------------------------------------------------------

def comfy_images_to_video_tensor(images: torch.Tensor) -> torch.Tensor:
    """
    ComfyUI supplies images as BHWC float32 [0,1].
    Returns BCFHW float32 [-1,1] ready for SparkVSR.
    B is treated as the frame count F.
    """
    # [F, H, W, C] → [F, C, H, W] → [1, C, F, H, W]
    x = images.permute(0, 3, 1, 2).contiguous()   # [F, C, H, W]
    x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)     # [1, C, F, H, W]
    # Scale [0,1] → [-1,1]
    x = x * 2.0 - 1.0
    return x


def video_tensor_to_comfy_images(video: torch.Tensor) -> torch.Tensor:
    """
    video: BCFHW float32 [0,1]  (output of SparkVSR)
    Returns BHWC float32 [0,1]  (ComfyUI IMAGE format)
    """
    # [B, C, F, H, W] → [F, C, H, W] → [F, H, W, C]
    x = video[0].permute(1, 2, 3, 0).contiguous()    # [F, H, W, C]
    x = x.clamp(0.0, 1.0).cpu().float()
    return x


# ---------------------------------------------------------------------------
# Core inference function (adapted from sparkvsr_inference_script.py)
# ---------------------------------------------------------------------------

def _get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    h, w = src
    r = h / w
    if r > (tgt_height / tgt_width):
        resize_height = tgt_height
        resize_width = int(round(tgt_height / h * w))
    else:
        resize_width = tgt_width
        resize_height = int(round(tgt_width / w * h))
    crop_top  = int(round((tgt_height - resize_height) / 2.0))
    crop_left = int(round((tgt_width  - resize_width)  / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def _prepare_rope(height, width, num_frames, transformer_config, vae_scale, device):
    if not _ROPE_OK:
        return None
    grid_height = height // (vae_scale * transformer_config.patch_size)
    grid_width  = width  // (vae_scale * transformer_config.patch_size)
    p   = transformer_config.patch_size
    p_t = transformer_config.patch_size_t
    base_size_w = transformer_config.sample_width  // p
    base_size_h = transformer_config.sample_height // p

    if p_t is None:
        crops = _get_resize_crop_region_for_grid((grid_height, grid_width), base_size_w, base_size_h)
        fc, fs = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=crops,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        base_num_frames = (num_frames + p_t - 1) // p_t
        fc, fs = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(max(base_size_h, grid_height), max(base_size_w, grid_width)),
            device=device,
        )
    return fc, fs


@torch.no_grad()
def _run_sparkvsr(
    pipe,
    video: torch.Tensor,          # [B, C, F, H, W]  float32 [-1, 1]
    ref_frames: List[torch.Tensor],
    ref_indices: List[int],
    chunk_start_idx: int,
    sr_noise_step: int,
    ref_guidance_scale: float,
    empty_prompt_embedding: Optional[torch.Tensor],
) -> torch.Tensor:
    """Single-pass forward through SparkVSR. Returns [B, C, F, H, W] in [0,1]."""
    video = video.to(pipe.device, dtype=pipe.dtype)

    # Encode LQ video
    latent_dist = pipe.vae.encode(video).latent_dist
    lq_latent   = latent_dist.sample() * pipe.vae.config.scaling_factor
    B, C, F, H, W = lq_latent.shape
    device = lq_latent.device
    dtype  = lq_latent.dtype

    # Build reference latent
    full_ref_latent = torch.zeros_like(lq_latent)
    for i, idx in enumerate(ref_indices):
        if i >= len(ref_frames):
            break
        local_idx = idx - chunk_start_idx
        lat_idx   = local_idx // 4
        if 0 <= lat_idx < F:
            rf = ref_frames[i].to(device, dtype=dtype)
            chunk = rf.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4, 1, 1)
            lat = pipe.vae.encode(chunk).latent_dist.sample() * pipe.vae.config.scaling_factor
            full_ref_latent[:, :, lat_idx, :, :] = lat[0, :, 0, :, :]

    # CFG
    do_cfg = abs(ref_guidance_scale - 1.0) > 1e-3
    if do_cfg:
        cond   = torch.cat([lq_latent, full_ref_latent], dim=1)
        uncond = torch.cat([lq_latent, torch.zeros_like(full_ref_latent)], dim=1)
        input_latent = torch.cat([uncond, cond], dim=0)
    else:
        input_latent = torch.cat([lq_latent, full_ref_latent], dim=1)

    # Patch-size-T padding
    p_t   = pipe.transformer.config.patch_size_t
    ncopy = 0
    if p_t is not None:
        ncopy = input_latent.shape[2] % p_t
        if ncopy:
            first = input_latent[:, :, :1, :, :]
            input_latent = torch.cat([first.repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)

    # Prompt embedding (use empty)
    if empty_prompt_embedding is not None:
        pe = empty_prompt_embedding.to(device, dtype=dtype)
        if pe.shape[0] != B:
            pe = pe.repeat(B, 1, 1)
    else:
        tok = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        pe = pipe.text_encoder(tok.input_ids.to(device))[0]
        _, sl, _ = pe.shape
        pe = pe.view(B, sl, -1).to(dtype=dtype)

    latents = input_latent.permute(0, 2, 1, 3, 4)   # [B, F, C, H, W]
    if do_cfg:
        pe = torch.cat([pe, pe], dim=0)

    timesteps = torch.full(
        (latents.shape[0],), fill_value=sr_noise_step,
        dtype=torch.long, device=device,
    )

    # RoPE
    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    rotary_emb = None
    if pipe.transformer.config.use_rotary_positional_embeddings and _ROPE_OK:
        rope = _prepare_rope(H * vae_scale, W * vae_scale, F,
                             pipe.transformer.config, vae_scale, device)
        if rope:
            rotary_emb = rope

    # OFS
    ofs = None
    if pipe.transformer.config.ofs_embed_dim is not None:
        ofs = torch.full((latents.shape[0],), fill_value=2.0, device=device, dtype=dtype)

    # Transformer forward
    pred = pipe.transformer(
        hidden_states=latents,
        encoder_hidden_states=pe,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        ofs=ofs,
        return_dict=False,
    )[0]

    pred_slice = pred[:, :, :16, :, :].transpose(1, 2)
    lq_sample  = latents[:, :, :16, :, :].transpose(1, 2)

    if do_cfg:
        u, c = pred_slice.chunk(2)
        pred_slice = u + ref_guidance_scale * (c - u)
        lq_sample  = lq_sample.chunk(2)[1]
        timesteps  = timesteps.chunk(2)[0]

    lat_gen = pipe.scheduler.get_velocity(pred_slice, lq_sample, timesteps)

    if p_t is not None and ncopy > 0:
        lat_gen = lat_gen[:, :, ncopy:, :, :]

    # Decode
    out = pipe.vae.decode(lat_gen / pipe.vae.config.scaling_factor).sample
    out = (out * 0.5 + 0.5).clamp(0.0, 1.0)
    return out


def _make_temporal_chunks(F: int, chunk_len: int, overlap_t: int):
    if chunk_len == 0:
        return [(0, F)]
    stride = chunk_len - overlap_t
    if stride <= 0:
        raise ValueError("chunk_len must be > overlap_t")
    starts = list(range(0, F - overlap_t, stride))
    if not starts or starts[-1] + chunk_len < F:
        starts.append(max(0, F - chunk_len))
    chunks = [(s, min(s + chunk_len, F)) for s in starts]
    # Deduplicate last chunk if it overlaps fully with previous
    if len(chunks) >= 2 and chunks[-1] == chunks[-2]:
        chunks.pop()
    return chunks


def _make_spatial_tiles(H: int, W: int, tile_h: int, tile_w: int, overlap_h: int, overlap_w: int):
    if tile_h == 0 or tile_w == 0:
        return [(0, H, 0, W)]
    stride_h = tile_h - overlap_h
    stride_w = tile_w - overlap_w
    hs = list(range(0, H - overlap_h, stride_h))
    if not hs or hs[-1] + tile_h < H:
        hs.append(max(0, H - tile_h))
    ws = list(range(0, W - overlap_w, stride_w))
    if not ws or ws[-1] + tile_w < W:
        ws.append(max(0, W - tile_w))
    tiles = []
    for h_s in hs:
        for w_s in ws:
            tiles.append((h_s, min(h_s + tile_h, H), w_s, min(w_s + tile_w, W)))
    return tiles


def _get_valid_region(t_s, t_e, h_s, h_e, w_s, w_e, F, H, W, ot, oh, ow):
    tl = t_e - t_s; hl = h_e - h_s; wl = w_e - w_s
    vts = 0 if t_s == 0 else ot // 2
    vte = tl if t_e == F else tl - ot // 2
    vhs = 0 if h_s == 0 else oh // 2
    vhe = hl if h_e == H else hl - oh // 2
    vws = 0 if w_s == 0 else ow // 2
    vwe = wl if w_e == W else wl - ow // 2
    return dict(vts=vts, vte=vte, vhs=vhs, vhe=vhe, vws=vws, vwe=vwe,
                ots=t_s+vts, ote=t_s+vte, ohs=h_s+vhs, ohe=h_s+vhe,
                ows=w_s+vws, owe=w_s+vwe)


# ===========================================================================
# ╔══════════════════════════════════════════════════════════╗
# ║  NODE 1 — SparkVSR_LoadPipeline                          ║
# ║  Equivalent of: SeedVR2 (Down)Load DiT M...              ║
# ╚══════════════════════════════════════════════════════════╝
# ===========================================================================

class SparkVSR_LoadPipeline:
    """
    Loads the SparkVSR pipeline (CogVideoX transformer + scheduler + tokenizer +
    text encoder).  The VAE is loaded here too but its tiling config is set by
    SparkVSR_LoadVAE.

    Equivalent of the 'SeedVR2 (Down)Load DiT Model' node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "checkpoints/sparkvsr-s2/ckpt-500-sft",
                        "tooltip": "Path to SparkVSR Stage-2 checkpoint folder.",
                    },
                ),
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {
                        "default": "bfloat16",
                        "tooltip": "Weight dtype. bfloat16 is native on RTX 40xx/50xx.",
                    },
                ),
                "device": (
                    "STRING",
                    {
                        "default": "cuda:0",
                        "tooltip": "Device to load the model on (cuda:0, cuda:1, cpu).",
                    },
                ),
                "offload_device": (
                    ["none", "cpu"],
                    {
                        "default": "cpu",
                        "tooltip": (
                            "none: keep everything on GPU (fastest, needs ~22 GB VRAM). "
                            "cpu: sequential CPU offload (~8 GB VRAM, slower)."
                        ),
                    },
                ),
                "cache_model": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Cache the loaded model in memory between runs.",
                    },
                ),
            }
        }

    RETURN_TYPES  = ("SPARKVSR_PIPELINE",)
    RETURN_NAMES  = ("pipeline",)
    FUNCTION      = "load"
    CATEGORY      = "SparkVSR"
    DESCRIPTION   = "Load SparkVSR pipeline (transformer + VAE + text encoder)."

    def load(
        self,
        model_path: str,
        dtype: str,
        device: str,
        offload_device: str,
        cache_model: bool,
    ):
        if not _DIFFUSERS_OK:
            raise RuntimeError(
                "diffusers is not installed. "
                "Run: pip install diffusers transformers accelerate"
            )

        # Normalise path (Windows backslash support)
        model_path = str(Path(model_path))

        cache_key = f"{model_path}|{dtype}|{device}|{offload_device}"
        if cache_model and cache_key in _PIPELINE_CACHE:
            logger.info(f"[SparkVSR] Returning cached pipeline: {cache_key}")
            return (_PIPELINE_CACHE[cache_key],)

        logger.info(f"[SparkVSR] Loading pipeline from {model_path} …")
        torch_dtype = _str_to_dtype(dtype)

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        if offload_device == "cpu":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        if cache_model:
            if cache_key in _PIPELINE_CACHE:
                del _PIPELINE_CACHE[cache_key]
            _PIPELINE_CACHE[cache_key] = pipe
            logger.info(f"[SparkVSR] Pipeline cached: {cache_key}")

        return (pipe,)


# ===========================================================================
# ╔══════════════════════════════════════════════════════════╗
# ║  NODE 2 — SparkVSR_LoadVAE                               ║
# ║  Equivalent of: SeedVR2 (Down)Load VAE M...              ║
# ╚══════════════════════════════════════════════════════════╝
# ===========================================================================

class SparkVSR_LoadVAE:
    """
    Configures (and optionally replaces) the VAE inside a loaded pipeline.
    Controls tiling for low-VRAM decode/encode.

    Equivalent of the 'SeedVR2 (Down)Load VAE Model' node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("SPARKVSR_PIPELINE",),
                # Encode settings
                "encode_tiled": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Tile VAE encoding (saves VRAM during encode)."},
                ),
                "encode_tile_size": (
                    "INT",
                    {"default": 704, "min": 64, "max": 2048, "step": 16,
                     "tooltip": "Spatial tile size for VAE encode (px)."},
                ),
                "encode_tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Overlap between encode tiles (px)."},
                ),
                # Decode settings
                "decode_tiled": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Tile VAE decoding (saves VRAM, recommended on 16 GB)."},
                ),
                "decode_tile_size": (
                    "INT",
                    {"default": 736, "min": 64, "max": 2048, "step": 16,
                     "tooltip": "Spatial tile size for VAE decode (px)."},
                ),
                "decode_tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Overlap between decode tiles (px)."},
                ),
                # Temporal slicing
                "enable_slicing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Frame-by-frame VAE processing (saves VRAM)."},
                ),
                "tile_debug": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print tile coordinates during processing."},
                ),
            }
        }

    RETURN_TYPES  = ("SPARKVSR_PIPELINE",)
    RETURN_NAMES  = ("pipeline",)
    FUNCTION      = "configure"
    CATEGORY      = "SparkVSR"
    DESCRIPTION   = "Configure VAE tiling / slicing for VRAM-efficient encode & decode."

    def configure(
        self,
        pipeline,
        encode_tiled: bool,
        encode_tile_size: int,
        encode_tile_overlap: int,
        decode_tiled: bool,
        decode_tile_size: int,
        decode_tile_overlap: int,
        enable_slicing: bool,
        tile_debug: bool,
    ):
        vae = pipeline.vae

        if enable_slicing:
            vae.enable_slicing()
        else:
            vae.disable_slicing()

        if decode_tiled or encode_tiled:
            vae.enable_tiling()
        else:
            vae.disable_tiling()

        # Store tiling config as attributes so the upscaler node can read them
        pipeline._sparkvsr_vae_cfg = {
            "encode_tiled":        encode_tiled,
            "encode_tile_size":    encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled":        decode_tiled,
            "decode_tile_size":    decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "tile_debug":          tile_debug,
        }

        if tile_debug:
            logger.info(f"[SparkVSR VAE] Config: {pipeline._sparkvsr_vae_cfg}")

        return (pipeline,)


# ===========================================================================
# ╔══════════════════════════════════════════════════════════╗
# ║  NODE 3 — SparkVSR_VideoUpscaler                         ║
# ║  Equivalent of: SeedVR2 Video Upscaler v2.5...           ║
# ╚══════════════════════════════════════════════════════════╝
# ===========================================================================

class SparkVSR_VideoUpscaler:
    """
    Main SparkVSR upscaling node.

    Inputs:
      image              — ComfyUI IMAGE (BHWC float32 [0,1]) treated as video frames
      pipeline           — Loaded pipeline from SparkVSR_LoadPipeline (+ VAE from LoadVAE)
      seed               — Random seed
      upscale            — Integer upscale factor (1–8)
      max_resolution     — Cap output at this pixel count along the longest edge (0 = no cap)
      batch_size         — Frames to process per temporal chunk (0 = all at once)
      uniform_batch_size — Round batch_size to match VAE temporal factor
      color_correction   — Color correction mode: none, lab, wavelet
      temporal_overlap   — Frame overlap between temporal chunks
      prepend_frames     — Duplicate first N frames before processing
      ref_mode           — Reference keyframe mode: no_ref, manual
      ref_indices        — Comma-separated keyframe indices for manual ref mode
      ref_guidance_scale — CFG scale for keyframe adherence (1.0 = no guidance)
      tile_size_h        — Spatial tile height (0 = no tiling)
      tile_size_w        — Spatial tile width  (0 = no tiling)
      overlap_h          — Spatial tile overlap (height)
      overlap_w          — Spatial tile overlap (width)
      sr_noise_step      — Noise step for single-step DDPM SR (default 399)
      offload_device     — Override offload device (none / cpu)
      enable_debug       — Print detailed timing / shape info

    Output: upscaled frames as ComfyUI IMAGE (BHWC float32 [0,1])
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline": ("SPARKVSR_PIPELINE",),
            },
            "optional": {
                "seed": (
                    "INT",
                    {"default": 313, "min": 0, "max": 0xFFFFFFFF,
                     "tooltip": "Random seed for reproducibility."},
                ),
                "upscale": (
                    "INT",
                    {"default": 4, "min": 1, "max": 8,
                     "tooltip": "Upscale factor (e.g. 4 = 4× super-resolution)."},
                ),
                "max_resolution": (
                    "INT",
                    {"default": 0, "min": 0, "max": 7680, "step": 16,
                     "tooltip": "Cap longest edge to this value (px). 0 = no cap."},
                ),
                "batch_size": (
                    "INT",
                    {"default": 81, "min": 1, "max": 513, "step": 8,
                     "tooltip": "Number of frames per temporal chunk. Lower = less VRAM."},
                ),
                "uniform_batch_size": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Round batch_size to nearest (8n+1) for VAE."},
                ),
                "color_correction": (
                    ["none", "lab", "wavelet"],
                    {"default": "lab", "tooltip": "Post-process color correction mode."},
                ),
                "temporal_overlap": (
                    "INT",
                    {"default": 8, "min": 0, "max": 64, "step": 1,
                     "tooltip": "Frame overlap between temporal chunks (blending zone)."},
                ),
                "prepend_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16,
                     "tooltip": "Duplicate first N frames before the sequence."},
                ),
                "ref_mode": (
                    ["no_ref", "manual"],
                    {"default": "no_ref",
                     "tooltip": (
                         "no_ref: blind SR (no keyframes). "
                         "manual: use ref_indices as keyframe hints."
                     )},
                ),
                "ref_indices": (
                    "STRING",
                    {"default": "0", "multiline": False,
                     "tooltip": "Comma-separated 0-based frame indices for manual ref mode."},
                ),
                "ref_guidance_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                     "tooltip": "CFG scale for keyframe adherence (1.0 = disabled)."},
                ),
                "tile_size_h": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 16,
                     "tooltip": "Spatial tile height in px (0 = no tiling)."},
                ),
                "tile_size_w": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 16,
                     "tooltip": "Spatial tile width in px (0 = no tiling)."},
                ),
                "overlap_h": (
                    "INT",
                    {"default": 32, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Spatial tile overlap height (px)."},
                ),
                "overlap_w": (
                    "INT",
                    {"default": 32, "min": 0, "max": 256, "step": 8,
                     "tooltip": "Spatial tile width overlap (px)."},
                ),
                "sr_noise_step": (
                    "INT",
                    {"default": 399, "min": 0, "max": 999,
                     "tooltip": "Single-step DDPM noise level (default 399)."},
                ),
                "enable_debug": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print shape and timing info to console."},
                ),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "upscale"
    CATEGORY      = "SparkVSR"
    DESCRIPTION   = "SparkVSR video super-resolution — processes ComfyUI image batch as video frames."

    # ------------------------------------------------------------------
    def upscale(
        self,
        image: torch.Tensor,
        pipeline,
        seed: int = 313,
        upscale: int = 4,
        max_resolution: int = 0,
        batch_size: int = 81,
        uniform_batch_size: bool = True,
        color_correction: str = "lab",
        temporal_overlap: int = 8,
        prepend_frames: int = 0,
        ref_mode: str = "no_ref",
        ref_indices: str = "0",
        ref_guidance_scale: float = 1.0,
        tile_size_h: int = 0,
        tile_size_w: int = 0,
        overlap_h: int = 32,
        overlap_w: int = 32,
        sr_noise_step: int = 399,
        enable_debug: bool = False,
    ) -> Tuple[torch.Tensor]:
        torch.manual_seed(seed)

        # ------ Determine device ------
        device = next(pipeline.transformer.parameters()).device
        if str(device) == "meta":
            # CPU offload active — run on cuda:0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if enable_debug:
            logger.info(f"[SparkVSR] Input image shape: {image.shape}  device={device}")

        # ------ Convert ComfyUI IMAGE → video tensor ------
        # image: [F, H, W, C] float32 [0,1]  (F = number of frames)
        F_orig, H_orig, W_orig, C = image.shape

        # Prepend frames
        if prepend_frames > 0:
            first_frame = image[:1].repeat(prepend_frames, 1, 1, 1)
            image = torch.cat([first_frame, image], dim=0)

        F_in = image.shape[0]

        # Build upscaled LR video (bilinear upscale of the input)
        # [F, H, W, C] → [F, C, H, W]
        video_lr = image.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]

        # Compute target resolution
        target_h = H_orig * upscale
        target_w = W_orig * upscale
        if max_resolution > 0:
            scale = max_resolution / max(target_h, target_w)
            if scale < 1.0:
                target_h = int(target_h * scale) // 16 * 16
                target_w = int(target_w * scale) // 16 * 16

        # Upscale LR to HR size
        video_up = F.interpolate(
            video_lr,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )  # [F, C, H, W]

        # Pad to multiples of 8 (VAE requirement)
        pad_h = (8 - target_h % 8) % 8
        pad_w = (8 - target_w % 8) % 8
        if pad_h or pad_w:
            video_up = F.pad(video_up, (0, pad_w, 0, pad_h))

        padded_h = target_h + pad_h
        padded_w = target_w + pad_w

        # Pad temporal to multiple of 8 frames
        pad_f = (8 - (F_in % 8)) % 8
        if pad_f:
            last = video_up[-1:].repeat(pad_f, 1, 1, 1)
            video_up = torch.cat([video_up, last], dim=0)

        F_padded = video_up.shape[0]

        # video_up is [0,1] (bilinear-interpolated from PIL input); scale to [-1,1]
        video_norm = video_up * 2.0 - 1.0   # [-1, 1]

        # Add batch dim: [1, C, F, H, W]
        video_batch = video_norm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

        if enable_debug:
            logger.info(f"[SparkVSR] Video tensor shape: {video_batch.shape}")

        # ------ Reference frames ------
        ref_idx_list: List[int] = []
        ref_frame_tensors: List[torch.Tensor] = []
        if ref_mode == "manual":
            try:
                ref_idx_list = sorted(set(int(x.strip()) for x in ref_indices.split(",") if x.strip()))
            except ValueError:
                logger.warning("[SparkVSR] Invalid ref_indices — falling back to no_ref.")
                ref_idx_list = []

            # For manual refs, use the upscaled frames as reference
            for idx in ref_idx_list:
                if idx < F_padded:
                    rf = video_norm[idx]  # [C, H, W] in [-1,1]
                    ref_frame_tensors.append(rf)

        # ------ Empty prompt embedding ------
        empty_prompt_embedding = None
        ep_path = _SPARKVSR_ROOT / "pretrained_weights" / "prompt_embeddings"
        ep_file = ep_path / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
        if ep_file.exists() and _SAFETENSORS_OK:
            try:
                empty_prompt_embedding = load_safetensors(str(ep_file))["prompt_embedding"]
            except Exception as e:
                logger.warning(f"[SparkVSR] Could not load empty prompt embedding: {e}")

        # ------ Chunk/tile processing ------
        chunk_len = batch_size
        if uniform_batch_size and chunk_len > 0:
            # Round down to nearest (8n+1) — VAE temporal requirement
            n = max(1, (chunk_len - 1) // 8)
            chunk_len = 8 * n + 1

        time_chunks   = _make_temporal_chunks(F_padded, chunk_len, temporal_overlap)
        spatial_tiles = _make_spatial_tiles(padded_h, padded_w, tile_size_h, tile_size_w,
                                            overlap_h, overlap_w)

        if enable_debug:
            logger.info(f"[SparkVSR] Chunks={len(time_chunks)} Tiles={len(spatial_tiles)}")

        output_video = torch.zeros_like(video_batch)

        for t_s, t_e in time_chunks:
            for h_s, h_e, w_s, w_e in spatial_tiles:
                chunk = video_batch[:, :, t_s:t_e, h_s:h_e, w_s:w_e]

                # Crop reference frames to spatial tile
                tile_refs = [rf[:, h_s:h_e, w_s:w_e] for rf in ref_frame_tensors]

                gen = _run_sparkvsr(
                    pipe=pipeline,
                    video=chunk,
                    ref_frames=tile_refs,
                    ref_indices=ref_idx_list,
                    chunk_start_idx=t_s,
                    sr_noise_step=sr_noise_step,
                    ref_guidance_scale=ref_guidance_scale,
                    empty_prompt_embedding=empty_prompt_embedding,
                )

                # Blend into output
                r = _get_valid_region(t_s, t_e, h_s, h_e, w_s, w_e,
                                      F_padded, padded_h, padded_w,
                                      temporal_overlap, overlap_h, overlap_w)
                output_video[:, :,
                             r["ots"]:r["ote"],
                             r["ohs"]:r["ohe"],
                             r["ows"]:r["owe"]] = gen[:, :,
                                                       r["vts"]:r["vte"],
                                                       r["vhs"]:r["vhe"],
                                                       r["vws"]:r["vwe"]]

        # ------ Remove padding ------
        if pad_f:
            output_video = output_video[:, :, :-pad_f, :, :]
        if pad_h:
            output_video = output_video[:, :, :, :-pad_h, :]
        if pad_w:
            output_video = output_video[:, :, :, :, :-pad_w]

        # Remove prepended frames
        if prepend_frames > 0:
            output_video = output_video[:, :, prepend_frames:, :, :]

        # ------ Color correction ------
        if color_correction != "none":
            output_video = _apply_color_correction(
                output_video,
                video_batch[:, :, :output_video.shape[2],
                            :output_video.shape[3], :output_video.shape[4]],
                mode=color_correction,
            )

        # ------ Convert to ComfyUI IMAGE ------
        result = video_tensor_to_comfy_images(output_video)   # [F, H, W, C]

        _free_memory()

        if enable_debug:
            logger.info(f"[SparkVSR] Output shape: {result.shape}")

        return (result,)


# ---------------------------------------------------------------------------
# Color correction utilities
# ---------------------------------------------------------------------------

def _apply_color_correction(
    output: torch.Tensor,    # [B, C, F, H, W] [0,1]
    reference: torch.Tensor, # [B, C, F, H, W] [-1,1]
    mode: str = "lab",
) -> torch.Tensor:
    """Match chrominance of `output` to `reference` using LAB or wavelet transfer."""
    ref_01 = (reference * 0.5 + 0.5).clamp(0, 1)
    if mode == "lab":
        return _lab_color_fix(output, ref_01)
    elif mode == "wavelet":
        return _wavelet_color_fix(output, ref_01)
    return output


def _rgb_to_lab_approx(x: torch.Tensor) -> torch.Tensor:
    """Very fast approximate sRGB → L*a*b* for color correction (no scipy needed)."""
    # sRGB → linear
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    # linear → XYZ (D65)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # Normalise by D65 white
    X /= 0.95047; Z /= 1.08883
    def f(t):
        delta = 6 / 29
        return torch.where(t > delta ** 3, t ** (1/3), t / (3 * delta**2) + 4/29)
    fX, fY, fZ = f(X), f(Y), f(Z)
    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b_ = 200 * (fY - fZ)
    return torch.cat([L, a, b_], dim=1)


def _lab_to_rgb_approx(lab: torch.Tensor) -> torch.Tensor:
    L, a, b_ = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fY = (L + 16) / 116
    fX = a / 500 + fY
    fZ = fY - b_ / 200
    delta = 6 / 29
    def finv(t):
        return torch.where(t > delta, t**3, 3 * delta**2 * (t - 4/29))
    X = finv(fX) * 0.95047
    Y = finv(fY)
    Z = finv(fZ) * 1.08883
    r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb_lin = torch.cat([r, g, b], dim=1).clamp(0, 1)
    # linear → sRGB
    return torch.where(rgb_lin > 0.0031308,
                       1.055 * rgb_lin ** (1/2.4) - 0.055,
                       12.92 * rgb_lin).clamp(0, 1)


def _lab_color_fix(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match luminance-normalised chrominance of output to reference."""
    B, C, F, H, W = output.shape
    out_frames = output[0].permute(1, 0, 2, 3)    # [F, C, H, W]
    ref_frames  = reference[0].permute(1, 0, 2, 3)
    corrected = []
    for i in range(F):
        o = out_frames[i:i+1]    # [1, C, H, W]
        r = ref_frames[i:i+1]
        o_lab = _rgb_to_lab_approx(o)
        r_lab = _rgb_to_lab_approx(r)
        # Transfer a,b channels (chrominance) from reference, keep L from output
        fixed_lab = torch.cat([o_lab[:, :1], r_lab[:, 1:]], dim=1)
        fixed_rgb = _lab_to_rgb_approx(fixed_lab)
        corrected.append(fixed_rgb)
    result = torch.stack(corrected, dim=2)  # [1, C, F, H, W]
    return result.clamp(0, 1)


def _wavelet_color_fix(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Simple high-frequency wavelet colour correction:
    keep high-freq from output, low-freq colour from reference.
    Uses a 2-level Haar-like box-filter approximation.
    """
    def _box_downsample(x, factor=2):
        return F.avg_pool2d(x, kernel_size=factor, stride=factor, padding=0)

    def _box_upsample(x, factor=2):
        return F.interpolate(x, scale_factor=factor, mode="nearest")

    B, C, Fv, H, W = output.shape
    out_flat = output[0].permute(1, 0, 2, 3).reshape(Fv * B, C, H, W)
    ref_flat = reference[0].permute(1, 0, 2, 3).reshape(Fv * B, C, H, W)

    # Low frequency from reference, high frequency from output
    low_out = _box_upsample(_box_downsample(out_flat))[:, :, :H, :W]
    low_ref = _box_upsample(_box_downsample(ref_flat))[:, :, :H, :W]
    result  = out_flat - low_out + low_ref
    result  = result.reshape(Fv, C, H, W).unsqueeze(0).permute(0, 2, 1, 3, 4)
    return result.clamp(0, 1)


# ===========================================================================
# ComfyUI Node Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "SparkVSR_LoadPipeline":  SparkVSR_LoadPipeline,
    "SparkVSR_LoadVAE":       SparkVSR_LoadVAE,
    "SparkVSR_VideoUpscaler": SparkVSR_VideoUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkVSR_LoadPipeline":  "SparkVSR Load Pipeline",
    "SparkVSR_LoadVAE":       "SparkVSR Configure VAE",
    "SparkVSR_VideoUpscaler": "SparkVSR Video Upscaler",
}
