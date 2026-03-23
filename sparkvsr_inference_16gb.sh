#!/usr/bin/env bash
# ============================================================
# SparkVSR — 16 GB VRAM Optimized Inference Script
# Target: RTX 50xx / RTX 40xx / any GPU with 16 GB VRAM
#
# Key optimizations enabled:
#   --is_cpu_offload   : Sequentially offloads pipeline components
#                        (transformer, VAE, text-encoder) to CPU
#                        when not in use, keeping peak VRAM low.
#   --is_vae_st        : Enables VAE slicing + tiling to decode
#                        large frames without OOM.
#   --chunk_len 49     : Process video in 49-frame temporal chunks
#                        (with 8-frame overlap) instead of all at once.
#   --tile_size_hw     : Spatially tiles the video; 480 854 covers
#                        a ~480p tile which keeps transformer VRAM low.
#   --dtype bfloat16   : Use BF16 precision (RTX 50xx series supports
#                        this natively; halves parameter storage vs FP32).
#
# Trade-off: slower than full-VRAM mode due to CPU↔GPU transfers.
# ============================================================

# ---- Model Paths ----
# Stage-2 is the recommended model for inference.
MODEL_PATH="checkpoints/sparkvsr-s2/ckpt-500-sft"

# ---- Shared 16 GB optimization flags ----
# --tile_size_hw 480 854: 16:9-ratio tiles (~480p per tile) that fit in 16 GB VRAM.
#   480 height × 854 width ≈ standard 16:9 aspect ratio (854/480 ≈ 1.78).
#   Adjust to 360 640 for GPUs with less headroom, or 720 1280 if VRAM allows.
VRAM_FLAGS="--is_cpu_offload --is_vae_st --dtype bfloat16 --chunk_len 49 --overlap_t 8 --tile_size_hw 480 854 --overlap_hw 32 32"

# ============================================================
# Mode 1 — No-Ref  (blind VSR, no keyframe input required)
# ============================================================
CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir  datasets/test/UDM10/LQ-Video \
    --model_path $MODEL_PATH \
    --output_path results/UDM10/no_ref_16gb \
    --gt_dir     datasets/test/UDM10/GT-Video \
    --ref_mode   no_ref \
    --ref_prompt_mode fixed \
    --ref_guidance_scale 1.0 \
    --upscale 4 \
    $VRAM_FLAGS

# ============================================================
# Mode 2 — API-Ref  (keyframes restored via fal-ai API)
# Requires a valid API key in finetune/utils/ref_utils.py
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
#     --input_dir  datasets/test/UDM10/LQ-Video \
#     --model_path $MODEL_PATH \
#     --output_path results/UDM10/api_ref_16gb \
#     --gt_dir     datasets/test/UDM10/GT-Video \
#     --ref_mode   api \
#     --ref_prompt_mode fixed \
#     --ref_guidance_scale 1.0 \
#     --ref_indices 0 \
#     --upscale 4 \
#     $VRAM_FLAGS

# ============================================================
# Mode 3 — PiSA-SR Ref  (keyframes via PiSA-SR open-source model)
# Update the pisa_* paths before running.
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
#     --input_dir  datasets/test/UDM10/LQ-Video \
#     --model_path $MODEL_PATH \
#     --output_path results/UDM10/pisa_ref_16gb \
#     --gt_dir     datasets/test/UDM10/GT-Video \
#     --ref_mode   pisasr \
#     --ref_guidance_scale 1.0 \
#     --ref_indices 0 \
#     --upscale 4 \
#     --pisa_python_executable "path/to/your/pisasr/conda/env/bin/python" \
#     --pisa_script_path       "path/to/your/PiSA-SR/test_pisasr.py" \
#     --pisa_sd_model_path     "path/to/your/PiSA-SR/preset/models/stable-diffusion-2-1-base" \
#     --pisa_chkpt_path        "path/to/your/PiSA-SR/preset/models/pisa_sr.pkl" \
#     --pisa_gpu "0" \
#     $VRAM_FLAGS
