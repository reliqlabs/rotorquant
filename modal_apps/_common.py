"""Shared Modal scaffolding for Leanstral-on-H100 jobs.

Centralizes the image build, volume names, and the HF→Modal model-prep step
so the individual apps stay focused. Both `calibrate_rotorquant` and
`baseline_fp8_eval` import from here.
"""

from __future__ import annotations

import modal


# Volumes — these are the team-agreed names from the planning thread.
LEANSTRAL_MODELS_VOL = modal.Volume.from_name("leanstral-models", create_if_missing=True)
ROTORQUANT_CALIB_VOL = modal.Volume.from_name("rotorquant-calibration", create_if_missing=True)

LEANSTRAL_MODELS_PATH = "/mnt/leanstral"
ROTORQUANT_CALIB_PATH = "/mnt/calibration"

# HF source layout inside the volume after `prepare_leanstral_hf` runs once.
HF_INTERMEDIATE_DIR = f"{LEANSTRAL_MODELS_PATH}/leanstral-2603-hf"
CONSOLIDATED_DIR = f"{LEANSTRAL_MODELS_PATH}/leanstral-2603-consolidated"


_REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


def build_image() -> modal.Image:
    """Image carrying torch + transformers (mistral4) + our rotorquant fork.

    The fork is mounted via `add_local_dir` rather than `git clone` so local
    edits propagate without an image cache bust. Keeps iteration fast.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install([
            # Pin transformers to a release that ships mistral4 (PR #44760).
            "transformers>=5.8.0",
            "torch==2.12.0",
            "torchvision",
            "huggingface_hub[hf_xet]>=0.27.0",
            "safetensors",
            "scipy",
            "numpy",
            "mistral_common>=1.10.0",
            "accelerate",  # for device_map="auto" multi-GPU loading
            "tiktoken",
        ])
        .env({"PYTHONPATH": "/opt/rotorquant"})
        # Local mount: every `modal run` ships the current working tree, so
        # bug fixes land in the next call with no rebuild needed. Excludes
        # caches and the volumes mount points so we don't ship gigabytes.
        .add_local_dir(
            _REPO_ROOT, "/opt/rotorquant",
            ignore=["__pycache__", "*.pyc", ".git", ".pytest_cache", ".mypy_cache"],
        )
    )


def prepare_hf_intermediate_if_missing():
    """Convert the consolidated Leanstral checkpoint to HF format on the volume.

    Idempotent: if weights AND tokenizer are already on the volume, skip.
    Three states it handles cleanly:
      1. Nothing on volume — full prep (download + convert + tokenizer).
      2. Weights present but tokenizer missing (e.g., from an earlier broken
         run) — write only the tokenizer/processor.
      3. Everything present — no-op.
    """
    import os

    has_weights = os.path.exists(f"{HF_INTERMEDIATE_DIR}/config.json")
    has_tokenizer = os.path.exists(f"{HF_INTERMEDIATE_DIR}/tokenizer.json")
    if has_weights and has_tokenizer:
        print(f"[prep] {HF_INTERMEDIATE_DIR} already prepared, skipping")
        return

    if has_weights and not has_tokenizer:
        print(f"[prep] weights present, tokenizer missing — writing tokenizer only")
        from modal_apps._convert_mistral4_weight_to_hf import (  # type: ignore
            convert_and_write_processor_and_tokenizer,
        )
        from transformers import AutoConfig
        from pathlib import Path as _Path
        config = AutoConfig.from_pretrained(HF_INTERMEDIATE_DIR)
        convert_and_write_processor_and_tokenizer(
            _Path(CONSOLIDATED_DIR), _Path(HF_INTERMEDIATE_DIR), config,
        )
        LEANSTRAL_MODELS_VOL.commit()
        print(f"[prep] tokenizer written")
        return

    print(f"[prep] downloading consolidated checkpoint to {CONSOLIDATED_DIR}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="mistralai/Leanstral-2603",
        local_dir=CONSOLIDATED_DIR,
        allow_patterns=["*.safetensors", "params.json", "tekken.json", "chat_template.jinja"],
    )

    # Run the streaming converter we wrote (committed in the fork).
    print(f"[prep] running streaming consolidated → HF converter")
    from modal_apps._convert_streaming import convert_streaming  # type: ignore
    from modal_apps._convert_mistral4_weight_to_hf import (  # type: ignore
        convert_and_write_processor_and_tokenizer,
    )

    config = convert_streaming(
        input_dir=CONSOLIDATED_DIR,
        output_dir=HF_INTERMEDIATE_DIR,
        max_position_embeddings=1_048_576,
        output_format="fp8",
        shard_size_gb=5.0,
    )
    # The streaming variant only writes weights + config; the tokenizer and
    # processor writing lived in the original main() entry. Call it manually.
    from pathlib import Path as _Path
    convert_and_write_processor_and_tokenizer(
        _Path(CONSOLIDATED_DIR), _Path(HF_INTERMEDIATE_DIR), config,
    )
    LEANSTRAL_MODELS_VOL.commit()
    print(f"[prep] done, HF intermediate at {HF_INTERMEDIATE_DIR}")
