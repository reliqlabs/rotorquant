"""Microbenchmark: iso_flash_decode vs (iso_decompress + standard SDPA).

Times the fully fused IsoQuant attention kernel against the equivalent
pure-MLX pipeline (decompress K and V, then mx.fast.scaled_dot_product_attention)
across a range of context lengths and bit widths. Useful for the README
"how fast" claim and for spotting regressions.

Times the warm-cache state (excludes first-call compile time).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import mlx.core as mx

sys.path.insert(0, ".")

from turboquant.mlx_fused_iso_attention import (
    iso_compress, iso_decompress, iso_flash_decode,
    make_random_quaternions, _ISO_CODEBOOKS,
)


def _bench_once(fn, *args, warmup=2, iters=10):
    """Run fn(*args) warmup times to compile/load, then time iters runs."""
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def _reference_attention(q, k_packed_flat, k_norms_flat, v_packed_flat, v_norms_flat,
                         centroids, q_L, q_R, scale, D, bits, B, H, T):
    K = iso_decompress(k_packed_flat, k_norms_flat, D, bits, q_L, q_R, centroids)
    V = iso_decompress(v_packed_flat, v_norms_flat, D, bits, q_L, q_R, centroids)
    K = K.reshape(B, H, T, D)
    V = V.reshape(B, H, T, D)
    return mx.fast.scaled_dot_product_attention(q, K, V, scale=scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--mode", choices=["full", "fast"], default="full")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--T", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    D = args.head_dim
    H = args.n_heads
    B = 1
    n_groups = D // 4
    q_L = make_random_quaternions(n_groups, seed=1)
    q_R = make_random_quaternions(n_groups, seed=2) if args.mode == "full" else None
    centroids = _ISO_CODEBOOKS[args.bits]
    scale = 1.0 / math.sqrt(D)

    print(f"# IsoQuant flash decode microbench")
    print(f"# bits={args.bits} mode={args.mode} head_dim={D} n_heads={H} iters={args.iters}")
    print(f"# device={mx.default_device()}")
    print(f"#")
    print(f"# {'T':>6}  {'flash (ms)':>12}  {'ref (ms)':>12}  {'speedup':>8}  {'savings KV (×)':>14}")

    for T in args.T:
        mx.random.seed(T)
        K = mx.random.normal((B * H * T, D)).astype(mx.float32)
        V = mx.random.normal((B * H * T, D)).astype(mx.float32)
        k_packed_flat, k_norms_flat = iso_compress(K, args.bits, q_L, q_R, centroids)
        v_packed_flat, v_norms_flat = iso_compress(V, args.bits, q_L, q_R, centroids)
        k_packed = k_packed_flat.reshape(B, H, T, -1)
        k_norms = k_norms_flat.reshape(B, H, T)
        v_packed = v_packed_flat.reshape(B, H, T, -1)
        v_norms = v_norms_flat.reshape(B, H, T)
        q = mx.random.normal((B, H, 1, D)).astype(mx.float32)

        flash_t = _bench_once(
            iso_flash_decode,
            q, k_packed, k_norms, v_packed, v_norms,
            centroids, q_L, q_R, scale, D, args.bits,
            iters=args.iters,
        )

        ref_t = _bench_once(
            _reference_attention,
            q, k_packed_flat, k_norms_flat, v_packed_flat, v_norms_flat,
            centroids, q_L, q_R, scale, D, args.bits, B, H, T,
            iters=args.iters,
        )

        speedup = ref_t / flash_t if flash_t > 0 else float("inf")

        # Compression ratio of K + V vs bf16 reference.
        packed_bytes = (k_packed.nbytes + k_norms.nbytes
                        + v_packed.nbytes + v_norms.nbytes)
        bf16_bytes = 2 * B * H * T * D * 2  # K + V × bf16
        ratio = bf16_bytes / packed_bytes

        print(f"  {T:>6}  {flash_t * 1000:>12.3f}  {ref_t * 1000:>12.3f}  "
              f"{speedup:>7.2f}x  {ratio:>13.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
