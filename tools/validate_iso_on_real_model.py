"""End-to-end IsoQuant validation against a real mlx-lm model's KV cache.

Loads a small mlx-lm model, runs a prefill on a representative prompt,
extracts the K cache from one layer, then runs `iso_compress` / `iso_decompress`
on those vectors and reports per-vector cosine similarity statistics.

This is the "step 2" sibling of the synthetic-tensor tests in
`tests/test_mlx_fused_iso_attention.py`: it confirms our pure-MLX path
works on actual learned K activations (not just `mx.random.normal`), at
both head_dim=64 (small text models) and ultimately head_dim=128 (Leanstral).

Usage:
    python tools/validate_iso_on_real_model.py [model_repo] [bits] [layer_idx]

Defaults: mlx-community/Qwen2.5-0.5B-Instruct-4bit, 3 bits, layer 0.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

# Make `turboquant` importable when running from repo root.
sys.path.insert(0, ".")

from turboquant.mlx_fused_iso_attention import (  # noqa: E402
    compute_codebooks,
    iso_compress,
    iso_decompress,
    make_random_quaternions,
)


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    layer_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print(f"[iso-val] loading {repo}", flush=True)
    t0 = time.time()
    from mlx_lm import load
    model, tokenizer = load(repo)
    print(f"[iso-val] load() in {time.time() - t0:.1f}s", flush=True)

    prompt = (
        "Lloyd-Max quantization optimally allocates centroids by minimizing "
        "the expected squared error over the source distribution. Combined "
        "with orthogonal rotation, this lets us quantize key/value vectors."
    )
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    print(f"[iso-val] prefill on {tokens.shape[1]} tokens", flush=True)

    # Build a fresh cache and run prefill.
    cache = [layer.attention.create_cache() if hasattr(layer, "attention") else None
             for layer in model.layers] if hasattr(model, "layers") else None
    # mlx-lm has its own make_*_cache helpers; the simpler path is to just call
    # the model with a fresh cache and let the model wire it.
    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except Exception:
        cache = None

    logits = model(tokens, cache=cache)
    mx.eval(logits)
    print(f"[iso-val] prefill done", flush=True)

    if cache is None or len(cache) <= layer_idx:
        print(f"[iso-val] no usable cache layer at idx {layer_idx}", flush=True)
        return 1

    layer_cache = cache[layer_idx]
    K = getattr(layer_cache, "keys", None)
    if K is None and hasattr(layer_cache, "state"):
        K = layer_cache.state[0]  # some cache impls store as (K, V) tuple
    if K is None:
        print(f"[iso-val] could not find K tensor in layer cache; type={type(layer_cache)}", flush=True)
        return 1

    print(f"[iso-val] K shape: {K.shape}, dtype: {K.dtype}", flush=True)

    # mlx-lm typically returns (B, n_kv_heads, seq_buffer, head_dim) where
    # `seq_buffer` is allocated in 256-token chunks; only the first
    # `prompt_len` slots are valid (the rest are zero-initialized).
    B, n_h, seq_buffer, d = K.shape
    if d % 4 != 0:
        print(f"[iso-val] head_dim {d} not a multiple of 4 — IsoQuant requires it", flush=True)
        return 1
    seq_actual = tokens.shape[1]
    K_valid = K[:, :, :seq_actual, :]
    K_flat = K_valid.astype(mx.float32).reshape(-1, d)
    print(f"[iso-val] sliced cache to {seq_actual} valid positions -> "
          f"{K_flat.shape[0]} vectors of dim {d}", flush=True)

    # Codebook is precomputed for d=128 by default; regenerate for the real d.
    centroids = compute_codebooks(d, bits_list=(bits,))[bits]
    print(f"[iso-val] using Lloyd-Max codebook for d={d}, bits={bits} "
          f"(range: [{mx.min(centroids).item():.4f}, {mx.max(centroids).item():.4f}])",
          flush=True)

    n_groups = d // 4
    q_L = make_random_quaternions(n_groups, seed=42)
    q_R = make_random_quaternions(n_groups, seed=43)

    # Run compress / decompress.
    t1 = time.time()
    packed, norms = iso_compress(K_flat, bits, q_L, q_R, centroids)
    K_hat = iso_decompress(packed, norms, d, bits, q_L, q_R, centroids)
    mx.eval(K_hat)
    print(f"[iso-val] iso roundtrip in {time.time() - t1:.3f}s", flush=True)

    # Per-vector cosine sim.
    dot = mx.sum(K_flat * K_hat, axis=-1)
    nx = mx.linalg.norm(K_flat, axis=-1)
    ny = mx.linalg.norm(K_hat, axis=-1)
    cos = dot / (nx * ny + 1e-8)
    mx.eval(cos)

    avg = mx.mean(cos).item()
    p_min = mx.min(cos).item()
    p_05 = mx.sort(cos)[max(0, int(0.05 * cos.size))].item()
    print("[iso-val] cosine similarity over real K vectors:", flush=True)
    print(f"           avg:    {avg:.4f}", flush=True)
    print(f"           p5:     {p_05:.4f}", flush=True)
    print(f"           min:    {p_min:.4f}", flush=True)

    # Size: how much did packing shrink it?
    raw_bytes = K_flat.size * 2  # bf16 default for KV cache
    pkd_bytes = packed.size * 4 + norms.size * 4
    ratio = raw_bytes / pkd_bytes if pkd_bytes else float("inf")
    print(f"[iso-val] compression: bf16={raw_bytes / 1e6:.2f}MB  "
          f"packed={pkd_bytes / 1e6:.2f}MB  ratio={ratio:.2f}x", flush=True)

    if avg < 0.90:
        print(f"[iso-val] WARNING: avg cosine {avg:.4f} < 0.90 — quality regressed", flush=True)
        return 2
    print("[iso-val] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
