"""Append HF_MODEL_CARD_ADDITION.md to each MLX Leanstral repo README.

Idempotent: looks for an anchor line in the existing README and only writes
if the section isn't already present. Run from the repo root with the rotorquant
fork's HF_MODEL_CARD_ADDITION.md in place. Requires `huggingface_hub` and an
upload-capable HF token in HF_TOKEN or ~/.cache/huggingface/token.

Usage:
    python tools/upload_iso_card_addition.py            # all 4 repos
    python tools/upload_iso_card_addition.py --dry-run  # print diffs only
    python tools/upload_iso_card_addition.py --repo mvid/Leanstral-2603-MLX-4bit
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

DEFAULT_REPOS = [
    "mvid/Leanstral-2603-MLX-3bit",
    "mvid/Leanstral-2603-MLX-4bit",
    "mvid/Leanstral-2603-MLX-5bit",
    "mvid/Leanstral-2603-MLX-6bit",
]

# Sentinel line we look for in the existing README to decide whether the
# section is already there. Matches the section heading in
# HF_MODEL_CARD_ADDITION.md. If the section is found, skip the upload.
ANCHOR = "## KV cache compression with IsoQuant"


def _read_addition() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    addition_path = repo_root / "HF_MODEL_CARD_ADDITION.md"
    if not addition_path.exists():
        raise FileNotFoundError(f"{addition_path} not found")
    return addition_path.read_text()


def _process_one(repo: str, addition: str, dry_run: bool, api) -> None:
    from huggingface_hub import hf_hub_download, upload_file

    print(f"\n=== {repo} ===", flush=True)
    try:
        readme_path = hf_hub_download(
            repo_id=repo, filename="README.md", repo_type="model",
        )
    except Exception as e:
        print(f"  [skip] could not pull README: {type(e).__name__}: {e}")
        return

    current = Path(readme_path).read_text()
    if ANCHOR in current:
        print(f"  [skip] anchor '{ANCHOR}' already present — no upload needed")
        return

    new = current.rstrip() + "\n\n" + addition.lstrip()
    print(f"  [diff] current={len(current)} chars, new={len(new)} chars (+{len(new) - len(current)})")

    if dry_run:
        print(f"  [dry-run] would upload README.md to {repo}")
        return

    # Write to a tempfile so upload_file gets a stable path.
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
        tmp.write(new)
        tmp_path = tmp.name

    try:
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="docs: append IsoQuant KV cache compression section",
        )
        print(f"  [done] uploaded README.md ({len(new)} chars)")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append",
                    help="Repo id to update (can pass multiple times). "
                         "Default: all four Leanstral MLX repos.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen without uploading.")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"[hf] authenticated as {whoami.get('name', '?')}")
    except Exception as e:
        print(f"[hf] not authenticated ({e}). Set HF_TOKEN or run `hf auth login`.",
              file=sys.stderr)
        return 1

    addition = _read_addition()
    print(f"[hf] addition: {len(addition)} chars from HF_MODEL_CARD_ADDITION.md")

    repos = args.repo or DEFAULT_REPOS
    for repo in repos:
        _process_one(repo, addition, args.dry_run, api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
