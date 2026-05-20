"""Tests for the IsoQuant-backed KVCache subclass."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


def _make_cache(head_dim=128, bits=3, mode="full"):
    from turboquant.iso_kv_cache import IsoKVCache
    from turboquant.mlx_fused_iso_attention import make_random_quaternions
    n_groups = head_dim // 4
    q_L = make_random_quaternions(n_groups, seed=1)
    q_R = make_random_quaternions(n_groups, seed=2) if mode == "full" else None
    return IsoKVCache(bits=bits, q_L=q_L, q_R=q_R, head_dim=head_dim)


def test_update_and_fetch_shapes():
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(0)
    K = mx.random.normal((1, 4, 16, 128))  # (B, H, T, D)
    V = mx.random.normal((1, 4, 16, 128))
    K_out, V_out = cache.update_and_fetch(K, V)
    mx.eval(K_out, V_out)
    assert K_out.shape == (1, 4, 16, 128)
    assert V_out.shape == (1, 4, 16, 128)
    assert cache.offset == 16


def test_incremental_append_preserves_history():
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(0)
    K1 = mx.random.normal((1, 4, 10, 128))
    V1 = mx.random.normal((1, 4, 10, 128))
    cache.update_and_fetch(K1, V1)
    K2 = mx.random.normal((1, 4, 5, 128))
    V2 = mx.random.normal((1, 4, 5, 128))
    K_out, V_out = cache.update_and_fetch(K2, V2)
    mx.eval(K_out, V_out)
    assert K_out.shape == (1, 4, 15, 128)
    assert cache.offset == 15


def test_reconstruction_quality_matches_pure_iso():
    """End-to-end via the cache should produce the same K_hat as calling
    iso_compress + iso_decompress directly (with the same rotors)."""
    from turboquant.mlx_fused_iso_attention import (
        iso_compress, iso_decompress, make_random_quaternions, _ISO_CODEBOOKS,
    )
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(42)
    K = mx.random.normal((1, 4, 8, 128))
    V = mx.random.normal((1, 4, 8, 128))
    K_out, _ = cache.update_and_fetch(K, V)

    # Reference: bypass the cache entirely.
    K_flat = K.reshape(-1, 128).astype(mx.float32)
    packed, norms = iso_compress(K_flat, 3, cache.q_L, cache.q_R, cache.centroids)
    K_ref = iso_decompress(packed, norms, 128, 3, cache.q_L, cache.q_R, cache.centroids)
    K_ref = K_ref.reshape(1, 4, 8, 128)
    mx.eval(K_out, K_ref)
    diff = mx.max(mx.abs(K_out.astype(mx.float32) - K_ref)).item()
    assert diff < 1e-5, f"cache vs direct iso path diverge by {diff:.3e}"


def test_memory_savings():
    """Packed K + V should be ~4× smaller than bf16 at 3-bit."""
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(0)
    # 1k tokens × 4 heads × 128 dim × 2 (K and V) × 2 bytes (bf16) = ~2 MB ref
    K = mx.random.normal((1, 4, 1024, 128))
    V = mx.random.normal((1, 4, 1024, 128))
    cache.update_and_fetch(K, V)
    bf16_bytes = 2 * 1024 * 4 * 128 * 2  # K + V at bf16
    packed_bytes = cache.memory_bytes()
    ratio = bf16_bytes / packed_bytes
    # Theoretical max at 3-bit: 16/3 ≈ 5.3×; norms+packing overhead drops this.
    assert ratio > 3.0, f"compression ratio {ratio:.2f}× below 3× — packing broken?"


def test_attend_matches_decompress_sdpa():
    """`IsoKVCache.attend` must match SDPA against the decompressed cache."""
    import math
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(99)
    K = mx.random.normal((1, 4, 64, 128))
    V = mx.random.normal((1, 4, 64, 128))
    cache.update_and_fetch(K, V)

    q = mx.random.normal((1, 4, 1, 128))
    scale = 1.0 / math.sqrt(128)

    K_dec, V_dec = cache.state
    ref = mx.fast.scaled_dot_product_attention(q, K_dec, V_dec, scale=scale)
    fused = cache.attend(q, scale)
    mx.eval(ref, fused)
    diff = mx.max(mx.abs(ref - fused)).item()
    assert diff < 5e-3, f"attend vs SDPA diff={diff:.3e}"


def test_attend_raises_before_update():
    cache = _make_cache(head_dim=128, bits=3)
    q = mx.random.normal((1, 4, 1, 128))
    with pytest.raises(RuntimeError, match="before update_and_fetch"):
        cache.attend(q, 1.0)


def test_load_rotors_factory(tmp_path):
    """Round-trip rotors.safetensors -> factory -> per-layer cache."""
    import os
    from safetensors.torch import save_file
    import torch
    from turboquant.iso_kv_cache import load_rotors_into_cache_factory
    from turboquant.mlx_fused_iso_attention import make_random_quaternions

    rotors = {}
    for li in (0, 8, 35):
        q_L = make_random_quaternions(32, seed=li)
        q_R = make_random_quaternions(32, seed=li + 100)
        rotors[f"layer_{li}.q_L"] = torch.from_numpy(__import__("numpy").asarray(q_L))
        rotors[f"layer_{li}.q_R"] = torch.from_numpy(__import__("numpy").asarray(q_R))
    path = tmp_path / "rotors.safetensors"
    save_file(rotors, str(path), metadata={"format": "pt"})

    factory = load_rotors_into_cache_factory(str(path), head_dim=128, bits=3)
    assert factory(0) is not None
    assert factory(1) is None  # not in rotors -> caller falls back
    assert factory(35) is not None
    cache = factory(8)
    K = mx.random.normal((1, 2, 4, 128))
    V = mx.random.normal((1, 2, 4, 128))
    K_out, _ = cache.update_and_fetch(K, V)
    mx.eval(K_out)
    assert K_out.shape == (1, 2, 4, 128)
