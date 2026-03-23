"""
SparkVSR ComfyUI Custom Nodes
SeedVR2.5-style interface for SparkVSR Video Super-Resolution

Nodes:
  - SparkVSR_LoadPipeline  : Load SparkVSR transformer + full pipeline
  - SparkVSR_LoadVAE       : Load / configure the VAE with tiling options
  - SparkVSR_VideoUpscaler : Main video upscaling node

Installation:
  Copy (or symlink) this folder into ComfyUI/custom_nodes/SparkVSR/
  and restart ComfyUI.

Windows 10 + RTX 5070 Ti (16 GB VRAM) tested defaults:
  dtype=bfloat16, offload_device=cpu, vae_decode_tiled=True
"""

from .sparkvsr_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
