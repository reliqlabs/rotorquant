"""Compare calibrated rotors vs random rotors on a real model's K cache.

After Modal calibration produces `rotors.safetensors` (one q_L (+ q_R) per
calibrated layer), this script downloads it locally and runs the actual
quality comparison: load a model, capture its K cache at the calibrated
layers, run IsoQuant compress/decompress with both the calibrated and a
random-seed rotor, report cosine similarity for each.

Usage:
    # 1) pull the rotors locally from the Modal volume
    modal volume get rotorquant-calibration \
        iso/default/bits3-full/rotors.safetensors ./rotors.safetensors

    # 2) run validation against any mlx-lm model
    python tools/validate_calibrated_rotors.py rotors.safetensors \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --bits 3 --mode full --layers 0,4,8

On the M5 against the real Leanstral MLX quants the layer ids should match
what the Modal job captured (defaults to every 4th layer + 35).
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

sys.path.insert(0, ".")

from turboquant.mlx_fused_iso_attention import (  # noqa: E402
    compute_codebooks,
    iso_compress,
    iso_decompress,
    make_random_quaternions,
)


def _load_rotors(path: str) -> dict[str, mx.array]:
    """Read rotors.safetensors -> dict of MLX arrays."""
    out: dict[str, mx.array] = {}
    try:
        # mx.load handles .safetensors directly.
        loaded = mx.load(path)
    except Exception:
        # Fallback via the safetensors package -> numpy -> mlx.
        from safetensors.numpy import load_file
        loaded = {k: mx.array(v) for k, v in load_file(path).items()}
    if isinstance(loaded, dict):
        out.update(loaded)
    return out


def _cos_stats(K: mx.array, K_hat: mx.array) -> dict:
    dot = mx.sum(K * K_hat, axis=-1)
    nx = mx.linalg.norm(K, axis=-1)
    ny = mx.linalg.norm(K_hat, axis=-1)
    cos = dot / (nx * ny + 1e-8)
    mx.eval(cos)
    sorted_cos = mx.sort(cos)
    p5_idx = max(0, int(0.05 * cos.size))
    return {
        "mean": mx.mean(cos).item(),
        "p05": sorted_cos[p5_idx].item(),
        "min": mx.min(cos).item(),
        "n": cos.size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rotors", help="path to rotors.safetensors")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--mode", choices=["full", "fast"], default="full")
    ap.add_argument("--layers", default="0",
                    help="comma-separated layer indices to validate")
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    print(f"[val] loading rotors from {args.rotors}", flush=True)
    rotors = _load_rotors(args.rotors)
    if not rotors:
        print("[val] empty rotors file — abort", flush=True)
        return 1
    print(f"[val]   {len(rotors)} tensors loaded "
          f"(layers covered: {sorted({k.split('.')[0].removeprefix('layer_') for k in rotors})})",
          flush=True)

    print(f"[val] loading model {args.model}", flush=True)
    t0 = time.time()
    from mlx_lm import load
    model, tokenizer = load(args.model)
    print(f"[val]   load() in {time.time() - t0:.1f}s", flush=True)

    prompt = args.prompt or (
        "Lloyd-Max quantization minimizes squared error over the source "
        "distribution. Combined with orthogonal rotation, this lets us "
        "quantize key/value vectors used in attention. We will calibrate "
        "rotor parameters on a representative corpus."
    )
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    print(f"[val] prefilling {tokens.shape[1]} tokens", flush=True)

    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except Exception:
        cache = None
    out = model(tokens, cache=cache)
    mx.eval(out)
    if cache is None:
        print("[val] no cache available — abort", flush=True)
        return 1

    layer_ids = [int(x) for x in args.layers.split(",")]
    seq_actual = tokens.shape[1]

    for li in layer_ids:
        if li >= len(cache):
            print(f"[val] layer {li}: out of range (model has {len(cache)} cache slots)",
                  flush=True)
            continue
        layer_cache = cache[li]
        K = getattr(layer_cache, "keys", None)
        if K is None and hasattr(layer_cache, "state"):
            K = layer_cache.state[0]
        if K is None:
            print(f"[val] layer {li}: no K in cache", flush=True)
            continue
        K = K[:, :, :seq_actual, :].astype(mx.float32)
        d = K.shape[-1]
        if d % 4 != 0:
            print(f"[val] layer {li}: head_dim {d} not div by 4 — skip", flush=True)
            continue
        K_flat = K.reshape(-1, d)
        centroids = compute_codebooks(d, bits_list=(args.bits,))[args.bits]

        # ── calibrated rotors ──
        q_L_key = f"layer_{li}.q_L"
        q_R_key = f"layer_{li}.q_R"
        if q_L_key not in rotors:
            print(f"[val] layer {li}: no calibrated rotors (looked for {q_L_key})",
                  flush=True)
            continue
        q_L_cal = rotors[q_L_key]
        q_R_cal = rotors.get(q_R_key) if args.mode == "full" else None

        packed, norms = iso_compress(K_flat, args.bits, q_L_cal, q_R_cal, centroids)
        K_hat_cal = iso_decompress(packed, norms, d, args.bits, q_L_cal, q_R_cal, centroids)
        cal_stats = _cos_stats(K_flat, K_hat_cal)

        # ── random rotors (baseline) ──
        rng_L = make_random_quaternions(d // 4, seed=12345)
        rng_R = make_random_quaternions(d // 4, seed=12346) if args.mode == "full" else None
        packed, norms = iso_compress(K_flat, args.bits, rng_L, rng_R, centroids)
        K_hat_rng = iso_decompress(packed, norms, d, args.bits, rng_L, rng_R, centroids)
        rng_stats = _cos_stats(K_flat, K_hat_rng)

        lift = cal_stats["mean"] - rng_stats["mean"]
        print(f"[val] layer {li:>2}  d={d}  n_vec={cal_stats['n']:>5}", flush=True)
        print(f"          calibrated:  mean={cal_stats['mean']:.4f}  "
              f"p5={cal_stats['p05']:.4f}  min={cal_stats['min']:.4f}", flush=True)
        print(f"          random   :  mean={rng_stats['mean']:.4f}  "
              f"p5={rng_stats['p05']:.4f}  min={rng_stats['min']:.4f}", flush=True)
        print(f"          calibration lift on mean cosine: {lift:+.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
