"""
SparkVSR — ComfyUI Custom Node
================================
Clone this repository directly into ComfyUI/custom_nodes/ and restart ComfyUI.

  git clone https://github.com/naxci1/SparkVSR ComfyUI/custom_nodes/SparkVSR

Three nodes are registered:
  • SparkVSR Load Pipeline  — loads the CogVideoX transformer + scheduler
  • SparkVSR Configure VAE  — configures VAE tiling / slicing options
  • SparkVSR Video Upscaler — runs the actual video super-resolution

CLI usage is also supported; see sparkvsr_inference_script.py.
"""

from .sparkvsr_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
