"""End-to-end smoke: hot-swap IsoKVCache into a real mlx-lm model.

Loads a small mlx-lm model, runs the same prefill twice — once with the
default KV cache, once with our IsoKVCache wired in per-layer — and
compares logits + top-1 agreement.

Confirms our whole IsoQuant stack glues together against a live model
on Apple Silicon. Not a quality benchmark — that needs Leanstral on the
M5 — but a tight correctness/integration check that runs in seconds.

Usage:
    python tools/end_to_end_iso_smoke.py \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --bits 3 --mode full
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

sys.path.insert(0, ".")

from turboquant.iso_kv_cache import IsoKVCache
from turboquant.mlx_fused_iso_attention import make_random_quaternions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--mode", choices=["full", "fast"], default="full")
    ap.add_argument("--prompt", default=(
        "Define a Lean 4 theorem that addition of natural numbers is "
        "commutative, and prove it by induction."
    ))
    args = ap.parse_args()

    print(f"[smoke] loading {args.model}", flush=True)
    from mlx_lm import load
    model, tokenizer = load(args.model)

    tokens = mx.array(tokenizer.encode(args.prompt))[None, :]
    seq_len = tokens.shape[1]
    print(f"[smoke] prefill on {seq_len} tokens", flush=True)

    # ── Baseline: stock KVCache via make_prompt_cache ───────────────────────
    from mlx_lm.models.cache import make_prompt_cache
    baseline_cache = make_prompt_cache(model)
    t0 = time.time()
    base_out = model(tokens, cache=baseline_cache)
    mx.eval(base_out)
    base_dt = time.time() - t0
    base_logits = base_out[0, -1]  # last position logits

    n_layers = len(baseline_cache)
    head_dim = None
    for layer_cache in baseline_cache:
        K = getattr(layer_cache, "keys", None)
        if K is None and hasattr(layer_cache, "state"):
            K = layer_cache.state[0]
        if K is not None and K.ndim == 4:
            head_dim = K.shape[-1]
            break
    if head_dim is None:
        print("[smoke] could not infer head_dim from baseline cache", flush=True)
        return 1
    assert head_dim % 4 == 0, f"head_dim {head_dim} not divisible by 4"

    print(f"[smoke] baseline prefill {base_dt:.2f}s; "
          f"n_layers={n_layers} head_dim={head_dim}", flush=True)

    # ── IsoKVCache: per-layer, all using the same random rotors ─────────────
    n_groups = head_dim // 4
    q_L = make_random_quaternions(n_groups, seed=42)
    q_R = make_random_quaternions(n_groups, seed=43) if args.mode == "full" else None

    iso_cache = [
        IsoKVCache(bits=args.bits, q_L=q_L, q_R=q_R, head_dim=head_dim)
        for _ in range(n_layers)
    ]

    t1 = time.time()
    iso_out = model(tokens, cache=iso_cache)
    mx.eval(iso_out)
    iso_dt = time.time() - t1
    iso_logits = iso_out[0, -1]

    print(f"[smoke] iso prefill {iso_dt:.2f}s "
          f"(slower because pure-MLX path, no fused metal kernels wired in)",
          flush=True)

    # ── Compare ────────────────────────────────────────────────────────────
    cos = (mx.sum(base_logits * iso_logits) /
           (mx.linalg.norm(base_logits) * mx.linalg.norm(iso_logits) + 1e-8))
    mx.eval(cos)

    base_top = mx.argsort(-base_logits)[:10]
    iso_top = mx.argsort(-iso_logits)[:10]
    mx.eval(base_top, iso_top)
    base_set = set(base_top.tolist())
    iso_set = set(iso_top.tolist())
    top10_agree = len(base_set & iso_set)
    base_top1 = int(base_top[0].item())
    iso_top1 = int(iso_top[0].item())

    print("[smoke] ── results ──", flush=True)
    print(f"           logits cosine sim:  {cos.item():.4f}", flush=True)
    print(f"           top-10 token overlap: {top10_agree}/10", flush=True)
    print(f"           top-1 match: baseline={base_top1!r} "
          f"({tokenizer.decode([base_top1])!r}) "
          f"vs iso={iso_top1!r} "
          f"({tokenizer.decode([iso_top1])!r})  "
          f"{'✓' if base_top1 == iso_top1 else '✗'}",
          flush=True)
    iso_bytes = sum(c.memory_bytes() for c in iso_cache)
    # Rough baseline: 2 (K+V) × n_layers × n_kv_heads × seq × head_dim × 2(bf16)
    base_bytes = 2 * n_layers * 2 * seq_len * head_dim * 2  # crude estimate
    print(f"           iso cache: {iso_bytes / 1024:.1f} KB total across "
          f"{n_layers} layers (rough bf16 baseline: {base_bytes / 1024:.1f} KB)",
          flush=True)

    return 0 if top10_agree >= 5 else 2


if __name__ == "__main__":
    sys.exit(main())
