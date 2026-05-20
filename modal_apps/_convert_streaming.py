"""Streaming consolidated → HF converter for Mistral4 / Leanstral.

Patches the stock convert_mistral4_weight_to_hf.py to avoid holding the full
state_dict in RAM:

  * Lazy per-tensor reads from the input shards via safetensors.safe_open.
  * Per-layer expert buffer: fuse and flush as soon as a layer's 128 experts
    are complete, then drop them from memory.
  * Output written incrementally in ~5 GB safetensors shards with a generated
    model.safetensors.index.json.
  * Config + tokenizer + processor written directly (no meta-model roundtrip).

Reuses helpers from the stock convert script imported as a sibling module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))

# When bundled inside the modal_apps package, the sibling module is named
# _convert_mistral4_weight_to_hf (leading underscore to mark it private).
try:
    from _convert_mistral4_weight_to_hf import (  # noqa: E402
        EXPERT_KEY_PATTERN,
        _descale_fp8_to_bf16,
        _fuse_experts_for_layer,
        _get_text_renamings,
        _get_vision_renamings,
        _maybe_permute_vision_rope,
        _read_json,
        _rename_key,
        convert_and_write_processor_and_tokenizer,
        convert_config,
    )
except ImportError:
    # Fallback for standalone use outside the modal_apps tree.
    from convert_mistral4_weight_to_hf import (  # noqa: E402
        EXPERT_KEY_PATTERN,
        _descale_fp8_to_bf16,
        _fuse_experts_for_layer,
        _get_text_renamings,
        _get_vision_renamings,
        _maybe_permute_vision_rope,
        _read_json,
        _rename_key,
        convert_and_write_processor_and_tokenizer,
        convert_config,
    )
from transformers import GenerationConfig, Mistral3Config


class ShardedSafetensorsWriter:
    """Writes tensors into rolling ~max_shard_bytes safetensors files.

    Tensors are cloned on write so we don't keep the underlying input mmap
    alive longer than needed.
    """

    def __init__(self, output_dir: Path, max_shard_bytes: int = 5 * (1024 ** 3)):
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.current: dict[str, torch.Tensor] = {}
        self.current_size = 0
        self.shard_idx = 0
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def write(self, key: str, tensor: torch.Tensor) -> None:
        assert key not in self.weight_map and key not in self.current, f"duplicate key {key}"
        tensor = tensor.detach().contiguous().clone()
        size = tensor.numel() * tensor.element_size()
        if self.current_size + size > self.max_shard_bytes and self.current:
            self._flush()
        self.current[key] = tensor
        self.current_size += size
        self.total_size += size

    def _flush(self) -> None:
        fname = f"_partial-{self.shard_idx:05d}.safetensors"
        path = self.output_dir / fname
        save_file(self.current, str(path), metadata={"format": "pt"})
        for key in self.current:
            self.weight_map[key] = fname
        self.current.clear()
        self.current_size = 0
        self.shard_idx += 1

    def finalize(self) -> None:
        if self.current:
            self._flush()
        total = self.shard_idx
        rename_map: dict[str, str] = {}
        for idx in range(total):
            old = f"_partial-{idx:05d}.safetensors"
            new = f"model-{idx + 1:05d}-of-{total:05d}.safetensors"
            (self.output_dir / old).rename(self.output_dir / new)
            rename_map[old] = new
        final_weight_map = {k: rename_map[v] for k, v in self.weight_map.items()}
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": final_weight_map,
        }
        with open(self.output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)


def convert_streaming(
    input_dir,
    output_dir,
    max_position_embeddings: int,
    output_format: str,
    shard_size_gb: float = 5.0,
):
    # Coerce strings -> Path so callers can pass either.
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    params = _read_json(input_dir / "params.json")
    is_vision = params.get("vision_encoder") is not None
    is_fp8_source = params.get("quantization", {}).get("qformat_weight") == "fp8_e4m3"
    output_fp8 = output_format == "fp8" and is_fp8_source
    output_bf16 = not output_fp8

    config = convert_config(params, max_position_embeddings, is_vision, output_fp8)
    text_config = config.text_config if isinstance(config, Mistral3Config) else config
    n_experts = text_config.n_routed_experts
    vision_config = config.vision_config if isinstance(config, Mistral3Config) else None

    model_prefix = "model.language_model" if is_vision else "model"
    text_renamings = _get_text_renamings(model_prefix)
    vision_renamings = _get_vision_renamings() if is_vision else []

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(output_dir, int(shard_size_gb * (1024 ** 3)))

    # Per-layer expert buffer: layer_idx -> {(param_type, suffix): {expert_idx: tensor}}
    layer_buf: dict[int, dict[tuple, dict[int, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    layer_count: dict[int, int] = defaultdict(int)

    # Each layer expects n_experts * 3 (w1/w2/w3) * (3 if FP8 else 1) tensors
    suffix_count = 3 if is_fp8_source else 1
    expected_per_layer = n_experts * 3 * suffix_count

    total_keys_seen: set[str] = set()
    completed_layers: set[int] = set()

    def maybe_flush_layer(layer_idx: int) -> None:
        if layer_count[layer_idx] != expected_per_layer:
            return
        # Reshape buffer to the dict shape _fuse_experts_for_layer expects.
        grouped = {
            (layer_idx, pt, sfx): exps for (pt, sfx), exps in layer_buf[layer_idx].items()
        }
        base = f"{model_prefix}.layers.{layer_idx}.mlp.experts"
        layer_result = _fuse_experts_for_layer(grouped, layer_idx, n_experts, base, output_fp8)
        for k, v in layer_result.items():
            writer.write(k, v)
        del layer_buf[layer_idx]
        del layer_count[layer_idx]
        completed_layers.add(layer_idx)
        print(f"  fused layer {layer_idx}", flush=True)

    shards = sorted(p for p in input_dir.iterdir() if p.suffix == ".safetensors")
    assert shards, f"No .safetensors files found in {input_dir}"

    for shard_path in shards:
        print(f"Processing shard: {shard_path.name}", flush=True)
        with safe_open(str(shard_path), framework="pt") as f:
            keys = list(f.keys())

            # First pass over this shard: load expert keys directly into per-layer buffers.
            for old_key in keys:
                match = EXPERT_KEY_PATTERN.match(old_key)
                if not match:
                    continue
                assert old_key not in total_keys_seen, f"dup key {old_key}"
                total_keys_seen.add(old_key)
                layer_idx = int(match[1])
                expert_idx = int(match[2])
                param_type = match[3]  # w1|w2|w3
                suffix = match[4]      # weight|qscale_weight|qscale_act
                tensor = f.get_tensor(old_key)
                layer_buf[layer_idx][(param_type, suffix)][expert_idx] = tensor
                layer_count[layer_idx] += 1
                # Don't fuse mid-shard — wait until shard finishes to avoid touching FP8 ops while we still hold the mmap.
                # (Fusion clones via writer anyway, so this is just clarity.)

            # Second pass: non-expert keys.
            for old_key in keys:
                if EXPERT_KEY_PATTERN.match(old_key):
                    continue
                assert old_key not in total_keys_seen, f"dup key {old_key}"
                total_keys_seen.add(old_key)

                tensor = f.get_tensor(old_key)

                if output_bf16 and is_fp8_source:
                    if old_key.endswith((".qscale_act", ".qscale_weight")):
                        continue
                    if old_key.endswith(".weight"):
                        scale_key = old_key.rsplit(".weight", 1)[0] + ".qscale_weight"
                        if scale_key in keys:
                            scale_tensor = f.get_tensor(scale_key)
                            tensor = _descale_fp8_to_bf16(tensor, scale_tensor)
                            del scale_tensor

                new_key = _rename_key(old_key, text_renamings, vision_renamings)
                if vision_config is not None and "vision_tower" in new_key:
                    tensor = _maybe_permute_vision_rope(new_key, tensor, vision_config)

                writer.write(new_key, tensor)
                del tensor

        # End-of-shard: try to fuse any layers that became complete.
        # Iterate over a snapshot of keys since we mutate during iteration.
        for layer_idx in list(layer_count.keys()):
            maybe_flush_layer(layer_idx)

    # After all shards, every layer should be complete.
    assert not layer_buf, f"Layers with incomplete expert buffers: {sorted(layer_buf.keys())}"
    n_text_layers = text_config.num_hidden_layers
    missing = set(range(n_text_layers)) - completed_layers
    assert not missing, f"Missing fused layers: {sorted(missing)}"

    if text_config.tie_word_embeddings:
        raise NotImplementedError("tie_word_embeddings=True not handled in streaming path")

    writer.finalize()

    config.save_pretrained(str(output_dir))
    GenerationConfig(
        bos_token_id=text_config.bos_token_id,
        eos_token_id=text_config.eos_token_id,
        pad_token_id=text_config.pad_token_id,
    ).save_pretrained(str(output_dir))

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Mistral4 → HF converter (low memory)")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max_position_embeddings", type=int, default=1_048_576)
    parser.add_argument("--output_format", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--shard_size_gb", type=float, default=5.0)
    args = parser.parse_args()

    config = convert_streaming(
        args.input_dir,
        args.output_dir,
        args.max_position_embeddings,
        args.output_format,
        args.shard_size_gb,
    )
    convert_and_write_processor_and_tokenizer(args.input_dir, args.output_dir, config)
    print(f"Done. Wrote HF model to {args.output_dir}")


if __name__ == "__main__":
    main()
