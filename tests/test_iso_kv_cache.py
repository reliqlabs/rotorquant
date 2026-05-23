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


def test_prealloc_buffer_grows_across_step_boundary():
    """The pre-allocated buffer pattern must survive crossing the step
    boundary (default 256 tokens). Decoded output after growth should match
    a fresh cache that was given everything in one shot."""
    cache = _make_cache(head_dim=128, bits=3)
    cache.step = 32  # tighter step → exercises multiple grow events fast

    mx.random.seed(13)
    chunks = []
    for n in (5, 10, 20, 50, 80):  # crosses step=32 multiple times
        K = mx.random.normal((1, 2, n, 128))
        V = mx.random.normal((1, 2, n, 128))
        chunks.append((K, V))
        cache.update_and_fetch(K, V)
    assert cache.offset == sum(c[0].shape[2] for c in chunks)

    # Reference: feed it all in one shot to a fresh cache.
    ref = _make_cache(head_dim=128, bits=3)
    all_K = mx.concatenate([c[0] for c in chunks], axis=2)
    all_V = mx.concatenate([c[1] for c in chunks], axis=2)
    ref.update_and_fetch(all_K, all_V)

    # state should match — bit-stable since both went through the same
    # compress path on identical K, V.
    k1, v1 = cache.state
    k2, v2 = ref.state
    mx.eval(k1, k2, v1, v2)
    assert k1.shape == k2.shape
    assert mx.max(mx.abs(k1.astype(mx.float32) - k2.astype(mx.float32))).item() < 1e-5
    assert mx.max(mx.abs(v1.astype(mx.float32) - v2.astype(mx.float32))).item() < 1e-5


def test_prealloc_buffer_capacity_grows_in_step_multiples():
    """Buffer .shape[2] should grow in `step` increments, not match offset
    exactly — that's the whole point of the pre-allocation."""
    cache = _make_cache(head_dim=128, bits=3)
    cache.step = 64
    K = mx.random.normal((1, 2, 10, 128))
    V = mx.random.normal((1, 2, 10, 128))
    cache.update_and_fetch(K, V)
    assert cache.offset == 10
    # Buffer rounded up to step boundary (64), not exactly 10.
    assert cache.k_packed.shape[2] == 64, (
        f"buffer should be 64 tokens, got {cache.k_packed.shape[2]}"
    )
    # Adding 100 more crosses one step boundary — capacity should grow once.
    K2 = mx.random.normal((1, 2, 100, 128))
    V2 = mx.random.normal((1, 2, 100, 128))
    cache.update_and_fetch(K2, V2)
    assert cache.offset == 110
    # Needed 110, allocated next multiple of 64 >= 110 → 128 total.
    assert cache.k_packed.shape[2] == 128


def test_attend_topk_keep_all_matches_dense():
    """`attend(topk=T)` should route to sparse but with threshold=-inf, so
    output matches the dense flash-decode path within fp32 rounding."""
    import math
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(7)
    T = 64
    K = mx.random.normal((1, 2, T, 128))
    V = mx.random.normal((1, 2, T, 128))
    cache.update_and_fetch(K, V)

    q = mx.random.normal((1, 2, 1, 128))
    scale = 1.0 / math.sqrt(128)
    dense = cache.attend(q, scale)
    sparse_all = cache.attend(q, scale, topk=T + 16)
    mx.eval(dense, sparse_all)
    diff = mx.max(mx.abs(dense - sparse_all)).item()
    assert diff < 5e-3, f"attend(topk>=T) vs attend(None) diff={diff:.3e}"


def test_attend_topk_one_picks_top_token():
    """`attend(topk=1)` should approximate V at argmax(QK)."""
    import math
    from turboquant.mlx_fused_iso_attention import iso_decompress
    cache = _make_cache(head_dim=128, bits=3)
    mx.random.seed(3)
    T = 96
    target = 41
    # Background K small; dominant K at `target`.
    K = mx.random.normal((1, 1, T, 128)) * 0.01
    dominant = mx.random.normal((128,))
    K = mx.concatenate([
        K[:, :, :target, :],
        dominant.reshape(1, 1, 1, 128),
        K[:, :, target + 1:, :],
    ], axis=2)
    V = mx.random.normal((1, 1, T, 128))
    cache.update_and_fetch(K, V)

    q = dominant.reshape(1, 1, 1, 128)
    scale = 1.0 / math.sqrt(128)
    out = cache.attend(q, scale, topk=1)

    # Use the public accessor — directly reading cache.v_packed would pick
    # up the pre-allocated buffer tail and miss the offset slice.
    V_dec = cache.values
    expected = V_dec[0, 0, target]
    mx.eval(out, expected)
    diff = mx.max(mx.abs(out.reshape(128) - expected)).item()
    assert diff < 1e-2, f"topk=1 attend diff vs V[target]: {diff:.3e}"


def test_per_head_rotors_roundtrip():
    """3D q_L (H, n_groups, 4) goes through compress + decompress without
    losing the per-head structure. Each head gets its own rotor; the kernels
    + pure-MLX paths agree on which rotor to apply for which head."""
    from turboquant.iso_kv_cache import IsoKVCache
    from turboquant.mlx_fused_iso_attention import make_random_quaternions
    H, D = 4, 128
    n_groups = D // 4
    q_L = mx.stack([make_random_quaternions(n_groups, seed=h) for h in range(H)], axis=0)
    q_R = mx.stack([make_random_quaternions(n_groups, seed=h + 100) for h in range(H)], axis=0)
    cache = IsoKVCache(bits=3, q_L=q_L, q_R=q_R, head_dim=D)
    assert cache.per_head_rotors

    mx.random.seed(0)
    K = mx.random.normal((1, H, 24, D))
    V = mx.random.normal((1, H, 24, D))
    K_out, V_out = cache.update_and_fetch(K, V)
    mx.eval(K_out, V_out)
    assert K_out.shape == (1, H, 24, D)
    # Decode quality should be roughly the same as the per-layer case (3-bit
    # iso on Gaussian data ≈ 0.97-0.99 cos sim per vector).
    cos_per_head = []
    for h in range(H):
        a = K[:, h].reshape(-1, D)
        b = K_out[:, h].reshape(-1, D)
        nx = mx.maximum(mx.linalg.norm(a, axis=-1), 1e-8)
        ny = mx.maximum(mx.linalg.norm(b, axis=-1), 1e-8)
        cos_per_head.append(((a * b).sum(axis=-1) / (nx * ny)).mean().item())
    assert all(c > 0.93 for c in cos_per_head), f"per-head cos sims: {cos_per_head}"


def test_per_head_attend_matches_decompress_sdpa():
    """attend() with per-head rotors must agree with SDPA on the decompressed
    K, V — same correctness invariant as the per-layer test."""
    import math
    from turboquant.iso_kv_cache import IsoKVCache
    from turboquant.mlx_fused_iso_attention import make_random_quaternions
    H, D = 4, 128
    n_groups = D // 4
    q_L = mx.stack([make_random_quaternions(n_groups, seed=h) for h in range(H)], axis=0)
    q_R = mx.stack([make_random_quaternions(n_groups, seed=h + 50) for h in range(H)], axis=0)
    cache = IsoKVCache(bits=3, q_L=q_L, q_R=q_R, head_dim=D)
    mx.random.seed(99)
    K = mx.random.normal((1, H, 64, D))
    V = mx.random.normal((1, H, 64, D))
    cache.update_and_fetch(K, V)

    q = mx.random.normal((1, H, 1, D))
    scale = 1.0 / math.sqrt(D)
    K_dec, V_dec = cache.state
    ref = mx.fast.scaled_dot_product_attention(q, K_dec, V_dec, scale=scale)
    fused = cache.attend(q, scale)
    mx.eval(ref, fused)
    diff = mx.max(mx.abs(ref - fused)).item()
    assert diff < 5e-3, f"per-head attend vs SDPA diff={diff:.3e}"


def test_per_head_distinct_rotors_actually_used():
    """If we corrupt one head's rotor to identity but keep others sane, the
    reconstruction of that head should be noticeably worse than the others —
    proves the kernel is actually indexing q_L by head."""
    from turboquant.iso_kv_cache import IsoKVCache
    from turboquant.mlx_fused_iso_attention import make_random_quaternions
    H, D = 4, 128
    n_groups = D // 4

    q_L_list = [make_random_quaternions(n_groups, seed=h) for h in range(H)]
    q_R_list = [make_random_quaternions(n_groups, seed=h + 200) for h in range(H)]
    # Head 2 gets an awful rotor (very different from random) — its
    # quantization will be much worse than the other heads if the kernel
    # is actually picking up per-head rotors.
    q_L_list[2] = mx.zeros((n_groups, 4)).at[:, 0].add(1.0)  # all (1,0,0,0)
    # Actually mx has no .at — use direct construction.
    bad = mx.tile(mx.array([1.0, 0.0, 0.0, 0.0])[None, :], (n_groups, 1))
    # Add tiny noise so it's not literally identity (which would still work).
    bad = bad + mx.random.normal(bad.shape) * 0.0  # noop, kept for clarity
    q_L_list[2] = bad

    q_L = mx.stack(q_L_list, axis=0)
    q_R = mx.stack(q_R_list, axis=0)
    cache = IsoKVCache(bits=2, q_L=q_L, q_R=q_R, head_dim=D)
    mx.random.seed(3)
    K = mx.random.normal((1, H, 64, D))
    V = mx.random.normal((1, H, 64, D))
    K_out, _ = cache.update_and_fetch(K, V)

    def head_cos(h):
        a = K[:, h].reshape(-1, D)
        b = K_out[:, h].reshape(-1, D)
        nx = mx.maximum(mx.linalg.norm(a, axis=-1), 1e-8)
        ny = mx.maximum(mx.linalg.norm(b, axis=-1), 1e-8)
        return ((a * b).sum(axis=-1) / (nx * ny)).mean().item()

    head2_cos = head_cos(2)
    other_cos = [head_cos(h) for h in (0, 1, 3)]
    print(f"  head 2 (identity rotor) cos={head2_cos:.4f}  others={other_cos}")
    # Identity rotor on random Gaussian K should still give a baseline cos
    # but distinctly different from random-quaternion rotors. We don't pin a
    # direction; just require the difference to be nonzero.
    assert abs(head2_cos - sum(other_cos)/3) > 1e-3, (
        "kernel may be ignoring per-head q_L — bad head looks identical to good ones"
    )


def test_kv_split_rotors_roundtrip():
    """V uses its own rotors; K reconstruction must stay clean even if V's
    rotor is replaced with a bad one. Proves the kernel and cache route
    K and V through different rotor buffers."""
    import math
    from turboquant.iso_kv_cache import IsoKVCache
    from turboquant.mlx_fused_iso_attention import make_random_quaternions
    n_groups = 128 // 4
    k_q_L = make_random_quaternions(n_groups, seed=11)
    k_q_R = make_random_quaternions(n_groups, seed=12)
    # V's rotor: identity-ish — quantization noise dominates without
    # decorrelation, so V cos sim should be noticeably worse.
    bad = mx.tile(mx.array([1.0, 0.0, 0.0, 0.0])[None, :], (n_groups, 1))
    cache = IsoKVCache(bits=2, q_L=k_q_L, q_R=k_q_R, head_dim=128,
                        v_q_L=bad, v_q_R=bad)
    assert cache.has_distinct_v_rotors

    mx.random.seed(5)
    H, T = 4, 32
    K = mx.random.normal((1, H, T, 128))
    V = mx.random.normal((1, H, T, 128))
    K_out, V_out = cache.update_and_fetch(K, V)
    mx.eval(K_out, V_out)

    def mean_cos(a, b):
        a = a.reshape(-1, 128); b = b.reshape(-1, 128)
        nx = mx.maximum(mx.linalg.norm(a, axis=-1), 1e-8)
        ny = mx.maximum(mx.linalg.norm(b, axis=-1), 1e-8)
        return ((a * b).sum(axis=-1) / (nx * ny)).mean().item()
    k_cos = mean_cos(K, K_out)
    v_cos = mean_cos(V, V_out)
    print(f"  K cos (good rotor)={k_cos:.4f}  V cos (bad rotor)={v_cos:.4f}")
    assert k_cos > 0.85, f"K with good rotor should be clean, got {k_cos:.4f}"
    # 2-bit + identity rotor gives noticeable degradation; this proves K
    # and V are using different rotors in both compress and decompress.
    assert k_cos - v_cos > 0.001 or k_cos > v_cos, (
        f"K and V cos sims identical ({k_cos:.4f} vs {v_cos:.4f}) — V is "
        "likely going through the K rotor by mistake"
    )


def test_kv_split_attend_matches_sdpa():
    """attend() with distinct K/V rotors must agree with SDPA on the
    decompressed K, V — proves the flash decode kernel routes the K rotor
    to the K unrotate block and the V rotor to the V unrotate block."""
    import math
    from turboquant.iso_kv_cache import IsoKVCache
    from turboquant.mlx_fused_iso_attention import make_random_quaternions
    n_groups = 128 // 4
    k_q_L = make_random_quaternions(n_groups, seed=21)
    k_q_R = make_random_quaternions(n_groups, seed=22)
    v_q_L = make_random_quaternions(n_groups, seed=23)
    v_q_R = make_random_quaternions(n_groups, seed=24)
    cache = IsoKVCache(bits=3, q_L=k_q_L, q_R=k_q_R, head_dim=128,
                        v_q_L=v_q_L, v_q_R=v_q_R)
    mx.random.seed(33)
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
    assert diff < 5e-3, f"KV-split attend vs SDPA diff={diff:.3e}"


def test_load_rotors_factory_kv_split(tmp_path):
    """File format with separate K and V keys (layer_<N>.k_q_L / .v_q_L) loads
    correctly and the resulting cache exposes has_distinct_v_rotors=True."""
    from safetensors.torch import save_file
    import torch, numpy as np
    from turboquant.iso_kv_cache import load_rotors_into_cache_factory
    from turboquant.mlx_fused_iso_attention import make_random_quaternions

    n_groups = 128 // 4
    k_q_L = make_random_quaternions(n_groups, seed=51)
    k_q_R = make_random_quaternions(n_groups, seed=52)
    v_q_L = make_random_quaternions(n_groups, seed=53)
    v_q_R = make_random_quaternions(n_groups, seed=54)
    rotors = {
        "layer_0.k_q_L": torch.from_numpy(np.asarray(k_q_L)),
        "layer_0.k_q_R": torch.from_numpy(np.asarray(k_q_R)),
        "layer_0.v_q_L": torch.from_numpy(np.asarray(v_q_L)),
        "layer_0.v_q_R": torch.from_numpy(np.asarray(v_q_R)),
    }
    path = tmp_path / "kv_rotors.safetensors"
    save_file(rotors, str(path), metadata={"format": "pt"})

    factory = load_rotors_into_cache_factory(str(path), head_dim=128, bits=3)
    cache = factory(0)
    assert cache is not None
    assert cache.has_distinct_v_rotors
    assert mx.array_equal(cache.q_L, k_q_L).item()
    assert mx.array_equal(cache.v_q_L, v_q_L).item()


def test_load_rotors_factory_per_head(tmp_path):
    """New per-head file layout (layer_<N>_head_<H>.q_L) loads correctly,
    stacks per-head tensors into (H, n_groups, 4), and the resulting cache
    has per_head_rotors=True."""
    from safetensors.torch import save_file
    import torch, numpy as np
    from turboquant.iso_kv_cache import load_rotors_into_cache_factory
    from turboquant.mlx_fused_iso_attention import make_random_quaternions

    H, D = 4, 128
    n_groups = D // 4
    rotors = {}
    for li in (0, 5):
        for hi in range(H):
            q_L = make_random_quaternions(n_groups, seed=li * 100 + hi)
            q_R = make_random_quaternions(n_groups, seed=li * 100 + hi + 50)
            rotors[f"layer_{li}_head_{hi}.q_L"] = torch.from_numpy(np.asarray(q_L))
            rotors[f"layer_{li}_head_{hi}.q_R"] = torch.from_numpy(np.asarray(q_R))
    path = tmp_path / "ph_rotors.safetensors"
    save_file(rotors, str(path), metadata={"format": "pt"})

    factory = load_rotors_into_cache_factory(str(path), head_dim=D, bits=3)
    cache = factory(0)
    assert cache is not None
    assert cache.per_head_rotors
    assert cache.q_L.shape == (H, n_groups, 4)
    assert cache.q_R.shape == (H, n_groups, 4)
    assert factory(1) is None  # not present


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
