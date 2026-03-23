"""
SparkVSR Model Auto-Downloader
-------------------------------
Safely downloads the SparkVSR pipeline components from HuggingFace when they
are not present locally.  Used by both the ComfyUI nodes and the CLI inference
script.

Model sources
  Base model:          zai-org/CogVideoX1.5-5B-I2V
  SparkVSR weights:    JiongzeYu/SparkVSR
  Prompt embeddings:   JiongzeYu/SparkVSR  (subfolder: prompt_embeddings)

Downloads are performed with the official huggingface_hub library so they
inherit all its safety features: HTTPS, SHA-256 file verification, resume
support, and atomic moves.

Windows 10 compatible: no symlinks required (local_dir_use_symlinks=False).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("SparkVSR.downloader")

# ---------------------------------------------------------------------------
# HuggingFace repo IDs
# ---------------------------------------------------------------------------

REPO_BASE_MODEL     = "zai-org/CogVideoX1.5-5B-I2V"
REPO_SPARKVSR       = "JiongzeYu/SparkVSR"

# ---------------------------------------------------------------------------
# Default local paths (relative to SparkVSR repo root)
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL_DIR  = "pretrained_weights/CogVideoX1.5-5B-I2V"
DEFAULT_SPARKVSR_DIR    = "checkpoints/sparkvsr-s2/ckpt-500-sft"
DEFAULT_PROMPT_EMB_DIR  = "pretrained_weights/prompt_embeddings"

# The SHA-256 of the empty string, used as the prompt embedding filename
_EMPTY_PROMPT_FILENAME = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ".safetensors"
)


def _try_import_hf_hub():
    """Return the huggingface_hub module or raise a clear RuntimeError."""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for automatic model downloading.\n"
            "Install it with:  pip install huggingface_hub"
        )


def _is_valid_checkpoint_dir(path: Path) -> bool:
    """
    Return True if `path` looks like a usable checkpoint directory.
    We consider it valid when it contains at least one of the common model
    files (config.json, model_index.json, *.safetensors, *.bin, *.gguf).
    """
    if not path.is_dir():
        return False
    sentinel_patterns = [
        "config.json",
        "model_index.json",
        "*.safetensors",
        "*.bin",
        "*.gguf",
    ]
    for pattern in sentinel_patterns:
        if list(path.glob(pattern)):
            return True
    return False


def _is_valid_pipeline_dir(path: Path) -> bool:
    """
    Return True only if `path` is a **complete** diffusers pipeline directory.

    A proper pipeline directory must have ``model_index.json`` at its root.
    Without it ``CogVideoXImageToVideoPipeline.from_pretrained`` falls back to
    treating the directory as a single-model checkpoint and looks for
    ``config.json`` in the root — causing the cryptic
    "Error no file named config.json found in directory …" error.

    Having only ``*.safetensors`` files (e.g. manually placed transformer
    weights) is *not* sufficient.
    """
    return path.is_dir() and (path / "model_index.json").is_file()


def _download_repo(
    repo_id: str,
    local_dir: Path,
    subfolder: Optional[str] = None,
    token: Optional[str] = None,
    hf_hub=None,
) -> None:
    """
    Download an entire HuggingFace repository (or subfolder) to `local_dir`.
    Uses snapshot_download for atomic, resumable, SHA-verified transfers.
    """
    hf = hf_hub or _try_import_hf_hub()
    local_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = dict(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,   # Windows 10 safe
        resume_download=True,
        token=token,
    )
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**"]

    logger.info(f"[SparkVSR] Downloading {repo_id} → {local_dir} …")
    hf.snapshot_download(**kwargs)
    logger.info(f"[SparkVSR] Download complete: {local_dir}")


def _download_single_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: Optional[str] = None,
    hf_hub=None,
) -> Path:
    """
    Download a single file from a HuggingFace repo.
    Returns the local path of the downloaded file.
    """
    hf = hf_hub or _try_import_hf_hub()
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[SparkVSR] Downloading {repo_id}/{filename} → {local_dir} …")
    dest = hf.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    logger.info(f"[SparkVSR] File ready: {dest}")
    return Path(dest)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def ensure_base_model(
    local_dir: Path,
    token: Optional[str] = None,
    hf_hub=None,
) -> None:
    """
    Ensure the CogVideoX1.5-5B-I2V base model is present at `local_dir`.
    Downloads from HuggingFace if not found.
    """
    if _is_valid_checkpoint_dir(local_dir):
        logger.info(f"[SparkVSR] Base model found at {local_dir}")
        return

    print(
        f"\n[SparkVSR] Base model not found at '{local_dir}'.\n"
        f"  Downloading '{REPO_BASE_MODEL}' from HuggingFace (~10 GB) …\n"
        f"  This only happens once.  Please wait.\n"
    )
    _download_repo(REPO_BASE_MODEL, local_dir, token=token, hf_hub=hf_hub)


def ensure_sparkvsr_weights(
    local_dir: Path,
    token: Optional[str] = None,
    hf_hub=None,
) -> None:
    """
    Ensure the SparkVSR Stage-2 checkpoint is present at `local_dir`.
    Downloads from HuggingFace if not found.

    Uses ``_is_valid_pipeline_dir`` (requires ``model_index.json``) rather than
    the looser ``_is_valid_checkpoint_dir``.  A directory that contains only
    manually-placed ``.safetensors`` weights is **not** a valid pipeline and
    will trigger a (re)download of the complete checkpoint.
    """
    if _is_valid_pipeline_dir(local_dir):
        logger.info(f"[SparkVSR] SparkVSR weights found at {local_dir}")
        return

    print(
        f"\n[SparkVSR] SparkVSR weights not found at '{local_dir}'.\n"
        f"  Downloading '{REPO_SPARKVSR}' from HuggingFace (~5 GB) …\n"
        f"  This only happens once.  Please wait.\n"
    )
    _download_repo(REPO_SPARKVSR, local_dir, token=token, hf_hub=hf_hub)


def ensure_prompt_embeddings(
    local_dir: Path,
    token: Optional[str] = None,
    hf_hub=None,
) -> None:
    """
    Ensure the pre-computed empty-prompt embedding exists at `local_dir`.
    The file is part of the JiongzeYu/SparkVSR repo under prompt_embeddings/.
    """
    target = local_dir / _EMPTY_PROMPT_FILENAME
    if target.is_file() and target.stat().st_size > 0:
        logger.info(f"[SparkVSR] Prompt embedding found at {target}")
        return

    print(
        f"\n[SparkVSR] Prompt embedding not found at '{target}'.\n"
        f"  Downloading from '{REPO_SPARKVSR}' …\n"
    )
    try:
        _download_single_file(
            repo_id=REPO_SPARKVSR,
            filename=f"prompt_embeddings/{_EMPTY_PROMPT_FILENAME}",
            local_dir=local_dir,
            token=token,
            hf_hub=hf_hub,
        )
    except Exception as e:
        # Non-fatal: inference works without it (will compute prompt on-the-fly)
        logger.warning(
            f"[SparkVSR] Could not download prompt embedding: {e}\n"
            "           Will compute embeddings at runtime (slightly slower)."
        )


def ensure_all_models(
    sparkvsr_dir: Path,
    base_model_dir: Optional[Path] = None,
    prompt_emb_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> None:
    """
    Convenience function: download all missing components.

    The SparkVSR pipeline is a merged checkpoint that already includes the
    adapted transformer weights merged on top of the base model.  When the
    model_path is a full merged checkpoint directory, only `sparkvsr_dir`
    needs to be downloaded.

    If `sparkvsr_dir` does NOT look like a full pipeline (no model_index.json),
    the base model is also fetched and merged paths are expected.
    """
    hf = _try_import_hf_hub()

    ensure_sparkvsr_weights(sparkvsr_dir, token=token, hf_hub=hf)

    # Check if sparkvsr_dir is a complete pipeline (has model_index.json)
    if not _is_valid_pipeline_dir(sparkvsr_dir):
        # SparkVSR checkpoint only has adapted weights — need base model
        if base_model_dir is not None:
            ensure_base_model(base_model_dir, token=token, hf_hub=hf)

    if prompt_emb_dir is not None:
        ensure_prompt_embeddings(prompt_emb_dir, token=token, hf_hub=hf)
