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


def build_image() -> modal.Image:
    """Image carrying torch + transformers (mistral4) + our rotorquant fork."""
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
        # Bring the calibration code itself onto the image so the Modal
        # function can `from turboquant.isoquant import IsoQuantMSE` etc.
        .run_commands(
            "git clone --depth 1 https://github.com/reliqlabs/rotorquant.git /opt/rotorquant"
        )
        .env({"PYTHONPATH": "/opt/rotorquant"})
    )


def prepare_hf_intermediate_if_missing():
    """Convert the consolidated Leanstral checkpoint to HF format on the volume.

    Idempotent: if `HF_INTERMEDIATE_DIR/config.json` already exists, skip.
    First run is ~20-40 min (consolidated → HF FP8 via streaming converter).
    Subsequent runs are no-ops since the volume persists.
    """
    import os
    import sys

    if os.path.exists(f"{HF_INTERMEDIATE_DIR}/config.json"):
        print(f"[prep] {HF_INTERMEDIATE_DIR} already prepared, skipping")
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
    sys.path.insert(0, "/opt/rotorquant/tools")  # if we put it there later
    # The converter lives in /tmp/lean-convert/ locally; for Modal we ship a
    # copy alongside the modal_apps tree.
    from modal_apps._convert_streaming import convert_streaming  # type: ignore

    convert_streaming(
        input_dir=CONSOLIDATED_DIR,
        output_dir=HF_INTERMEDIATE_DIR,
        max_position_embeddings=1_048_576,
        output_format="fp8",
        shard_size_gb=5.0,
    )
    print(f"[prep] done, HF intermediate at {HF_INTERMEDIATE_DIR}")
