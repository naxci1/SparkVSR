#!/usr/bin/env python3
"""
convert_to_gguf.py — Convert SparkVSR / CogVideoX model weights to GGUF format.

GGUF (GGML Universal File) allows you to store model weights with optional
quantization, reducing VRAM requirements for inference.

Supported quantization types:
    f32    — Full precision float32 (no size reduction, baseline)
    f16    — Float16 (~50 % smaller than f32)
    bf16   — BFloat16 (~50 % smaller, native on RTX 40xx/50xx)
    q8_0   — 8-bit quantization (~75 % smaller, near-lossless quality)
    q4_0   — 4-bit quantization (~87.5 % smaller, slight quality drop)

Usage:
    # Convert the SparkVSR Stage-2 transformer to 8-bit GGUF
    # (auto-detects the transformer/ sub-folder from the pipeline root)
    python convert_to_gguf.py \\
        --model_dir  checkpoints/sparkvsr-s2/ckpt-500-sft \\
        --subfolder  transformer \\
        --output     sparkvsr_transformer_q8_0.gguf \\
        --quant_type q8_0

    # Convert the text encoder to 8-bit GGUF
    python convert_to_gguf.py \\
        --model_dir  checkpoints/sparkvsr-s2/ckpt-500-sft \\
        --subfolder  text_encoder \\
        --output     sparkvsr_text_encoder_q8_0.gguf \\
        --quant_type q8_0

    # You can also point directly to any sub-folder (no --subfolder needed):
    python convert_to_gguf.py \\
        --model_dir  checkpoints/sparkvsr-s2/ckpt-500-sft/transformer \\
        --output     sparkvsr_transformer_q8_0.gguf \\
        --quant_type q8_0

    # Convert the base CogVideoX transformer
    python convert_to_gguf.py \\
        --model_dir  pretrained_weights/CogVideoX1.5-5B-I2V \\
        --subfolder  transformer \\
        --output     cogvideox_transformer_f16.gguf \\
        --quant_type f16

Requirements:
    pip install gguf safetensors torch

Notes:
    - Sharded safetensors (e.g. diffusion_pytorch_model-00001-of-00005.safetensors)
      are merged automatically — no manual pre-merging is required.
    - Each pipeline component (transformer, text_encoder, …) should be
      converted separately and produces its own GGUF file.
    - After conversion, use the GGUF file with a compatible inference
      engine (e.g. ComfyUI + ComfyUI-GGUF extension) or load it back
      via the loader below for custom pipelines.
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load all safetensors / pytorch bin files in a directory
# ---------------------------------------------------------------------------

def _resolve_weight_dir(model_dir: Path, subfolder: Optional[str] = None) -> Path:
    """
    Return the directory that actually contains the weight files.

    When a full diffusers pipeline is downloaded with snapshot_download the
    individual component weights live in sub-folders (e.g. ``transformer/``,
    ``text_encoder/``) rather than at the pipeline root.  This function
    transparently handles all layouts:

      • ``--subfolder`` provided → use ``<model_dir>/<subfolder>/`` directly
        (if it has weight files) or fall back to ``model_dir`` itself.
      • Already pointing at a directory with weight files → returned as-is.
      • Full pipeline dir → tries ``transformer/`` as default subfolder.
    """
    _HAS_WEIGHTS = lambda d: (
        bool(list(d.glob("*.safetensors"))) or bool(list(d.glob("pytorch_model*.bin")))
    )

    # Explicit subfolder requested via --subfolder flag.
    if subfolder:
        sub_dir = model_dir / subfolder
        if sub_dir.is_dir() and _HAS_WEIGHTS(sub_dir):
            logger.info(f"  Using sub-folder: {sub_dir}")
            return sub_dir
        # subfolder dir exists but has no weights — let caller emit error.
        if sub_dir.is_dir():
            return sub_dir
        # subfolder doesn't exist at all
        logger.error(
            f"Sub-folder '{subfolder}' not found inside: {model_dir}\n"
            f"  Available sub-folders: "
            + ", ".join(d.name for d in model_dir.iterdir() if d.is_dir())
        )
        sys.exit(1)

    # Already pointing directly at weight files — use as-is.
    if _HAS_WEIGHTS(model_dir):
        return model_dir

    # Full pipeline layout: try the transformer/ sub-folder as default.
    transformer_dir = model_dir / "transformer"
    if transformer_dir.is_dir() and _HAS_WEIGHTS(transformer_dir):
        logger.info(
            f"  Detected full pipeline directory — defaulting to transformer "
            f"sub-folder: {transformer_dir}\n"
            f"  Tip: use --subfolder text_encoder to convert the text encoder instead."
        )
        return transformer_dir

    # Nothing found — return original so the caller can emit a clear error.
    return model_dir


def _load_state_dict(model_dir: Path, subfolder: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load all weights from a model directory into CPU numpy arrays.

    Handles both single-file and sharded safetensors layouts automatically.
    All shards matching ``*.safetensors`` are loaded and merged into a single
    dict — no manual pre-merging is required.
    """
    model_dir = _resolve_weight_dir(model_dir, subfolder=subfolder)
    state: Dict[str, np.ndarray] = {}

    # Prefer safetensors (faster, safer)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if st_files:
        try:
            from safetensors import safe_open
        except ImportError:
            logger.error("safetensors is not installed. Run: pip install safetensors")
            sys.exit(1)
        for st_path in st_files:
            logger.info(f"  Loading {st_path.name} …")
            with safe_open(str(st_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key).float().numpy()
                    state[key] = tensor
        return state

    # Fall back to pytorch bin files
    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if bin_files:
        for bin_path in bin_files:
            logger.info(f"  Loading {bin_path.name} …")
            loaded = torch.load(str(bin_path), map_location="cpu")
            for key, val in loaded.items():
                state[key] = val.float().numpy()
        return state

    logger.error(
        f"No .safetensors or .bin weight files found in: {model_dir}\n"
        "  • If you downloaded the full pipeline, specify which component to convert:\n"
        "      --subfolder transformer    (video transformer)\n"
        "      --subfolder text_encoder   (text encoder)\n"
        "  • Or point --model_dir directly to the sub-folder that contains the weights.\n"
        "  • Re-download with:\n"
        "      huggingface-cli download JiongzeYu/SparkVSR "
        "--local-dir checkpoints/sparkvsr-s2/ckpt-500-sft"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Quantization helpers (matching GGML on-disk format)
# ---------------------------------------------------------------------------

def _quantize_q8_0(data: np.ndarray) -> np.ndarray:
    """Quantize a float32 array to GGML Q8_0 blocks (32 elements/block)."""
    BLOCK_SIZE = 32
    flat = data.flatten().astype(np.float32)
    # Pad to multiple of BLOCK_SIZE
    pad = (BLOCK_SIZE - len(flat) % BLOCK_SIZE) % BLOCK_SIZE
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, BLOCK_SIZE)
    scales = np.abs(blocks).max(axis=1, keepdims=True) / 127.0
    scales = np.where(scales == 0, 1e-9, scales)
    quant = np.round(blocks / scales).astype(np.int8).clip(-127, 127)
    # Pack: each block = 2 bytes (scale as f16) + 32 bytes (int8)
    buf = bytearray()
    scale_f16 = scales[:, 0].astype(np.float16)
    for i in range(len(scale_f16)):
        buf += struct.pack("<e", float(scale_f16[i]))
        buf += quant[i].tobytes()
    return np.frombuffer(bytes(buf), dtype=np.uint8)


def _quantize_q4_0(data: np.ndarray) -> np.ndarray:
    """Quantize a float32 array to GGML Q4_0 blocks (32 elements/block)."""
    BLOCK_SIZE = 32
    flat = data.flatten().astype(np.float32)
    pad = (BLOCK_SIZE - len(flat) % BLOCK_SIZE) % BLOCK_SIZE
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, BLOCK_SIZE)
    scales = np.abs(blocks).max(axis=1, keepdims=True) / 7.0
    scales = np.where(scales == 0, 1e-9, scales)
    quant = np.round(blocks / scales).astype(np.int8).clip(-7, 7)
    # Pack two 4-bit values per byte.
    # GGML Q4_0 uses a bias of 8: values in [-7, 7] become [1, 15].
    # Value 0 is reserved as a sentinel; zero blocks are safely represented
    # because `scales` is clamped to ≥ 1e-9 above (never truly zero).
    quant_uint = (quant + 8).astype(np.uint8)  # shift to [1, 15]; 0 is reserved
    packed = quant_uint[:, 0::2] | (quant_uint[:, 1::2] << 4)
    buf = bytearray()
    scale_f16 = scales[:, 0].astype(np.float16)
    for i in range(len(scale_f16)):
        buf += struct.pack("<e", float(scale_f16[i]))
        buf += packed[i].tobytes()
    return np.frombuffer(bytes(buf), dtype=np.uint8)


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3

# GGML data-type enum values (matching ggml.h)
GGML_TYPE = {
    "f32":  0,
    "f16":  1,
    "q4_0": 2,
    "q8_0": 8,
    "bf16": 30,
}


def write_gguf(
    output_path: Path,
    state_dict: Dict[str, np.ndarray],
    quant_type: str,
    arch: str = "sparkvsr",
) -> None:
    """Write a GGUF file from a numpy state dict."""

    qt = quant_type.lower()
    if qt not in GGML_TYPE:
        raise ValueError(f"Unknown quant_type '{qt}'. Choose from: {list(GGML_TYPE)}")

    tensor_names: List[str] = sorted(state_dict.keys())
    n_tensors = len(tensor_names)

    # ------- Build metadata (key-value pairs) -------
    kv_items: List[tuple] = [
        ("general.architecture", "str", arch),
        ("general.quantization_version", "uint32", 2),
        ("general.quant_type", "str", qt),
        ("general.tensor_count", "uint64", n_tensors),
    ]

    kv_section = bytearray()
    for kkey, ktype, kval in kv_items:
        encoded_key = kkey.encode("utf-8")
        kv_section += struct.pack("<Q", len(encoded_key))
        kv_section += encoded_key
        if ktype == "str":
            kv_section += struct.pack("<I", 8)  # GGUF_TYPE_STRING = 8
            encoded_val = kval.encode("utf-8")
            kv_section += struct.pack("<Q", len(encoded_val))
            kv_section += encoded_val
        elif ktype == "uint32":
            kv_section += struct.pack("<I", 5)  # GGUF_TYPE_UINT32 = 5
            kv_section += struct.pack("<I", kval)
        elif ktype == "uint64":
            kv_section += struct.pack("<I", 7)  # GGUF_TYPE_UINT64 = 7
            kv_section += struct.pack("<Q", kval)

    n_kv = len(kv_items)

    # ------- Quantize / convert tensors -------
    logger.info(f"Quantizing {n_tensors} tensors to {qt} …")
    tensor_data_list: List[np.ndarray] = []
    tensor_info_list: List[tuple] = []  # (name, shape, ggml_type, offset)

    current_offset = 0
    for name in tensor_names:
        raw = state_dict[name].astype(np.float32)
        shape = raw.shape
        n_dims = len(shape)

        if qt == "f32":
            data = raw
            dtype_id = GGML_TYPE["f32"]
        elif qt == "f16":
            data = raw.astype(np.float16).view(np.uint8)
            dtype_id = GGML_TYPE["f16"]
        elif qt == "bf16":
            # BF16: upper 2 bytes of float32
            data = raw.view(np.uint32) >> 16
            data = data.astype(np.uint16).view(np.uint8)
            dtype_id = GGML_TYPE["bf16"]
        elif qt == "q8_0":
            data = _quantize_q8_0(raw)
            dtype_id = GGML_TYPE["q8_0"]
        elif qt == "q4_0":
            data = _quantize_q4_0(raw)
            dtype_id = GGML_TYPE["q4_0"]
        else:
            raise RuntimeError(f"Unhandled quant type: {qt}")

        tensor_data_list.append(data)
        tensor_info_list.append((name, shape, dtype_id, current_offset))
        current_offset += data.nbytes

    # ------- Build tensor-info section -------
    ti_section = bytearray()
    for name, shape, dtype_id, offset in tensor_info_list:
        encoded_name = name.encode("utf-8")
        ti_section += struct.pack("<Q", len(encoded_name))
        ti_section += encoded_name
        ti_section += struct.pack("<I", len(shape))          # n_dims
        for dim in reversed(shape):                          # GGUF uses column-major order
            ti_section += struct.pack("<Q", dim)
        ti_section += struct.pack("<I", dtype_id)            # ggml_type
        ti_section += struct.pack("<Q", offset)              # data offset

    # ------- Write GGUF file -------
    logger.info(f"Writing GGUF file: {output_path}")
    with open(output_path, "wb") as fout:
        # Header
        fout.write(GGUF_MAGIC)
        fout.write(struct.pack("<I", GGUF_VERSION))
        fout.write(struct.pack("<Q", n_tensors))
        fout.write(struct.pack("<Q", n_kv))
        # KV metadata
        fout.write(bytes(kv_section))
        # Tensor info
        fout.write(bytes(ti_section))
        # Alignment padding (GGUF aligns tensor data to 32 bytes)
        header_end = fout.tell()
        align = 32
        pad = (align - header_end % align) % align
        fout.write(b"\x00" * pad)
        # Tensor data
        for data in tensor_data_list:
            fout.write(data.tobytes())

    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info(f"Done. GGUF file size: {size_mb:.1f} MB  →  {output_path}")


# ---------------------------------------------------------------------------
# Model-file identification helper
# ---------------------------------------------------------------------------

MODEL_FILES = {
    "pretrained_weights/CogVideoX1.5-5B-I2V": {
        "description": "Base model (CogVideoX1.5-5B-I2V)",
        "components": {
            "transformer/": "Video transformer — main 5 B-param model (~10 GB BF16)",
            "vae/":         "Variational Autoencoder — encodes/decodes video frames (~1 GB)",
            "text_encoder/":"T5 text encoder — encodes text prompts (~4 GB)",
            "tokenizer/":   "Text tokenizer (no weights, just config files)",
            "scheduler/":   "Noise scheduler config (no weights)",
        }
    },
    "checkpoints/sparkvsr-s1/ckpt-10000-sft": {
        "description": "SparkVSR Stage-1 fine-tuned transformer weights",
        "components": {
            "*.safetensors": "Full transformer weights (same architecture as base model)",
        }
    },
    "checkpoints/sparkvsr-s2/ckpt-500-sft": {
        "description": "SparkVSR Stage-2 final weights (recommended for inference)",
        "components": {
            "*.safetensors": "Full transformer weights (same architecture as base model)",
        }
    },
}


def print_model_files() -> None:
    print("\n=== SparkVSR Model File Map ===\n")
    for path, info in MODEL_FILES.items():
        print(f"📂 {path}/")
        print(f"   {info['description']}")
        for comp, desc in info["components"].items():
            print(f"   ├── {comp:30s} {desc}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SparkVSR / CogVideoX model weights to GGUF format."
    )
    parser.add_argument(
        "--model_dir", type=Path, required=False, default=None,
        help=(
            "Path to the model directory. Accepts either a full pipeline "
            "directory (e.g. checkpoints/sparkvsr-s2/ckpt-500-sft) "
            "or a component sub-folder directly "
            "(e.g. checkpoints/sparkvsr-s2/ckpt-500-sft/transformer). "
            "Use --subfolder to select a component when pointing at the pipeline root."
        ),
    )
    parser.add_argument(
        "--subfolder", type=str, default=None,
        metavar="NAME",
        help=(
            "Component sub-folder to convert when --model_dir is a full pipeline "
            "directory.  Common values: 'transformer', 'text_encoder'. "
            "Defaults to 'transformer' when omitted and no weight files exist "
            "at the pipeline root."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output GGUF file path (default: <model_dir_name>_<quant>.gguf).",
    )
    parser.add_argument(
        "--quant_type", type=str, default="q8_0",
        choices=list(GGML_TYPE.keys()),
        help="Quantization type (default: q8_0).",
    )
    parser.add_argument(
        "--list_models", action="store_true",
        help="Print a map of all SparkVSR model files and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print_model_files()
        return

    if args.model_dir is None:
        logger.error("--model_dir is required unless --list_models is specified.")
        sys.exit(1)

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    output = args.output or (
        Path.cwd() / f"{model_dir.name}_{args.quant_type}.gguf"
    )

    logger.info(f"Model directory : {model_dir}")
    logger.info(f"Quantization    : {args.quant_type}")
    logger.info(f"Output path     : {output}")

    logger.info("Loading weights …")
    state_dict = _load_state_dict(model_dir, subfolder=args.subfolder)
    logger.info(f"Loaded {len(state_dict)} tensors.")

    write_gguf(output, state_dict, args.quant_type)


if __name__ == "__main__":
    main()
