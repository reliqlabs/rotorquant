"""Tests for MLX IsoQuant (quaternion 4D block) — pure-MLX path.

These run on any host with MLX; on non-Apple boxes MLX falls back to CPU.
The torch-parity test additionally requires the reference impl in
`turboquant.isoquant`.
"""

from __future__ import annotations

import math

import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


# ── Quaternion primitives ───────────────────────────────────────────────────


def test_quat_conj_involution():
    from turboquant.mlx_fused_iso_attention import _quat_conj
    mx.random.seed(0)
    q = mx.random.normal((8, 4))
    back = _quat_conj(_quat_conj(q))
    mx.eval(back)
    assert mx.max(mx.abs(q - back)).item() < 1e-6


def test_quat_mul_identity():
    """Multiplying by the identity quaternion (1,0,0,0) is a no-op."""
    from turboquant.mlx_fused_iso_attention import _quat_mul
    mx.random.seed(0)
    q = mx.random.normal((8, 4))
    ident = mx.broadcast_to(mx.array([1.0, 0.0, 0.0, 0.0]), q.shape)
    left = _quat_mul(ident, q)
    right = _quat_mul(q, ident)
    mx.eval(left, right)
    assert mx.max(mx.abs(left - q)).item() < 1e-6
    assert mx.max(mx.abs(right - q)).item() < 1e-6


def test_quat_mul_unit_is_unit():
    """Product of two unit quaternions is unit (norm preserved)."""
    from turboquant.mlx_fused_iso_attention import _quat_mul, make_random_quaternions
    a = make_random_quaternions(16, seed=1)
    b = make_random_quaternions(16, seed=2)
    c = _quat_mul(a, b)
    norms = mx.linalg.norm(c, axis=-1)
    mx.eval(norms)
    assert mx.max(mx.abs(norms - 1.0)).item() < 1e-5


# ── Forward / inverse roundtrip ─────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["full", "fast"])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_iso_rotate_unrotate_roundtrip(mode, d):
    from turboquant.mlx_fused_iso_attention import iso_rotate, iso_unrotate, make_random_quaternions
    mx.random.seed(42)
    n_groups = d // 4
    q_L = make_random_quaternions(n_groups, seed=1)
    q_R = make_random_quaternions(n_groups, seed=2) if mode == "full" else None

    x = mx.random.normal((4, d))
    back = iso_unrotate(iso_rotate(x, q_L, q_R), q_L, q_R)
    mx.eval(back)
    diff = mx.max(mx.abs(x - back)).item()
    # Quaternion-sandwich rotation should round-trip to ~floating-point precision.
    assert diff < 1e-4, f"roundtrip diff={diff:.3e} (mode={mode}, d={d})"


def test_iso_rotate_norm_preserving():
    """Unit quaternion sandwich preserves L2 norm of 4D blocks."""
    from turboquant.mlx_fused_iso_attention import iso_rotate, make_random_quaternions
    mx.random.seed(0)
    d = 128
    n_groups = d // 4
    q_L = make_random_quaternions(n_groups, seed=10)
    q_R = make_random_quaternions(n_groups, seed=11)

    x = mx.random.normal((8, d))
    y = iso_rotate(x, q_L, q_R)
    # Norms per 4-block should match.
    x_blocks = x.reshape(8, n_groups, 4)
    y_blocks = y.reshape(8, n_groups, 4)
    nx = mx.linalg.norm(x_blocks, axis=-1)
    ny = mx.linalg.norm(y_blocks, axis=-1)
    mx.eval(nx, ny)
    assert mx.max(mx.abs(nx - ny)).item() < 1e-4


# ── Pack/unpack roundtrip (shared format) ──────────────────────────────────


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [64, 128, 160])
def test_pack_unpack_roundtrip(bits, d):
    from turboquant.mlx_fused_iso_attention import _pack, _unpack
    mx.random.seed(0)
    n_levels = 1 << bits
    # mx.random.randint(low, high, shape) gives int values in [low, high).
    indices = mx.random.randint(0, n_levels, shape=(4, d)).astype(mx.uint32)
    packed = _pack(indices, bits)
    back = _unpack(packed, bits, d)
    mx.eval(back)
    assert mx.array_equal(indices.astype(mx.int32), back.astype(mx.int32)).item()


# ── Compress / decompress quality ──────────────────────────────────────────


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("mode", ["full", "fast"])
def test_iso_compress_decompress_cosine(bits, mode):
    """Quantization round-trip should preserve direction at ≥ 0.95 cosine sim."""
    from turboquant.mlx_fused_iso_attention import (
        iso_compress, iso_decompress, make_random_quaternions,
    )
    mx.random.seed(0)
    d = 128
    n_groups = d // 4
    q_L = make_random_quaternions(n_groups, seed=1)
    q_R = make_random_quaternions(n_groups, seed=2) if mode == "full" else None

    x = mx.random.normal((32, d))
    packed, norms = iso_compress(x, bits, q_L, q_R)
    x_hat = iso_decompress(packed, norms, d, bits, q_L, q_R)

    dot = mx.sum(x * x_hat, axis=-1)
    nx = mx.linalg.norm(x, axis=-1)
    ny = mx.linalg.norm(x_hat, axis=-1)
    cos = dot / (nx * ny + 1e-8)
    mx.eval(cos)
    avg = mx.mean(cos).item()
    # Loose bound for the MVP — sharper bound comes once fused kernels land.
    assert avg > 0.95, f"avg cosine={avg:.4f} too low (bits={bits}, mode={mode})"


# ── Parity with torch IsoQuantMSE ──────────────────────────────────────────


def test_iso_parity_with_torch_reference():
    """Pure-MLX iso pipeline should produce the same x_hat as the torch reference
    (up to fp32 rounding) when given the same quaternions and codebook."""
    torch = pytest.importorskip("torch")
    from turboquant.isoquant import IsoQuantMSE
    from turboquant.mlx_fused_iso_attention import (
        iso_compress, iso_decompress, _ISO_CODEBOOKS,
    )

    torch.manual_seed(0)
    mx.random.seed(0)
    d, bits = 128, 3

    iso = IsoQuantMSE(d, bits, mode='full', seed=42)
    x_t = torch.randn(8, d, dtype=torch.float32)
    x_hat_t, _ = iso(x_t)

    # Mirror torch's quaternions + codebook into MLX.
    q_L = mx.array(iso.q_L.detach().numpy())
    q_R = mx.array(iso.q_R.detach().numpy())
    centroids = mx.array(iso.centroids.detach().numpy())

    x = mx.array(x_t.numpy())
    packed, norms = iso_compress(x, bits, q_L, q_R, centroids)
    x_hat = iso_decompress(packed, norms, d, bits, q_L, q_R, centroids)
    mx.eval(x_hat)

    x_hat_torch_as_mx = mx.array(x_hat_t.detach().numpy())
    diff = mx.max(mx.abs(x_hat - x_hat_torch_as_mx)).item()
    assert diff < 1e-3, f"torch-parity diff={diff:.3e} (bits={bits}, mode=full)"


# ── PlanarQuant helpers (PR #8 backfill) ───────────────────────────────────


def test_planar_rotate_unrotate_roundtrip():
    from turboquant.mlx_fused_planar_attention import _planar_rotate, _planar_unrotate
    mx.random.seed(7)
    x = mx.random.normal((4, 128))
    back = _planar_unrotate(_planar_rotate(x))
    mx.eval(back)
    assert mx.max(mx.abs(x - back)).item() < 1e-5


def test_planar_compress_decompress_cosine():
    from turboquant.mlx_fused_planar_attention import (
        _compress, _decompress, _planar_rotate, _planar_unrotate,
    )
    mx.random.seed(0)
    d, bits = 128, 3
    x = mx.random.normal((32, d))
    packed, norms = _compress(x, bits, _planar_rotate)
    x_hat = _decompress(packed, norms, d, bits, _planar_unrotate, mx.float32)
    dot = mx.sum(x * x_hat, axis=-1)
    nx = mx.linalg.norm(x, axis=-1)
    ny = mx.linalg.norm(x_hat, axis=-1)
    cos = dot / (nx * ny + 1e-8)
    mx.eval(cos)
    avg = mx.mean(cos).item()
    assert avg > 0.95, f"planar avg cosine={avg:.4f}"


# ── Metal kernel parity (step E foundation) ────────────────────────────────


@pytest.mark.parametrize("mode", ["full", "fast"])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_iso_unrotate_metal_matches_mlx(mode, d):
    """Metal-fused iso_unrotate must match the pure-MLX iso_unrotate elementwise."""
    from turboquant.mlx_fused_iso_attention import (
        iso_unrotate, iso_unrotate_metal, make_random_quaternions,
    )
    mx.random.seed(0)
    n_groups = d // 4
    q_L = make_random_quaternions(n_groups, seed=7)
    q_R = make_random_quaternions(n_groups, seed=8) if mode == "full" else None

    x = mx.random.normal((4, d))
    ref = iso_unrotate(x, q_L, q_R)
    metal_out = iso_unrotate_metal(x, q_L, q_R)
    mx.eval(ref, metal_out)
    diff = mx.max(mx.abs(ref - metal_out)).item()
    assert diff < 1e-4, f"Metal vs MLX iso_unrotate diff={diff:.3e} (mode={mode}, d={d})"


def test_iso_metal_full_roundtrip():
    """forward MLX iso_rotate + Metal iso_unrotate_metal should round-trip."""
    from turboquant.mlx_fused_iso_attention import (
        iso_rotate, iso_unrotate_metal, make_random_quaternions,
    )
    mx.random.seed(0)
    d = 128
    n_groups = d // 4
    q_L = make_random_quaternions(n_groups, seed=11)
    q_R = make_random_quaternions(n_groups, seed=12)

    x = mx.random.normal((6, d))
    back = iso_unrotate_metal(iso_rotate(x, q_L, q_R), q_L, q_R)
    mx.eval(back)
    diff = mx.max(mx.abs(x - back)).item()
    assert diff < 1e-3, f"full-cycle roundtrip diff={diff:.3e}"


# ── Fused QK kernel correctness ────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["full", "fast"])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_iso_fused_qk_matches_reference(mode, bits):
    """Fused QK score should match `iso_decompress + matmul` within float-32 noise."""
    import math
    from turboquant.mlx_fused_iso_attention import (
        iso_compress, iso_decompress, iso_fused_qk_scores,
        make_random_quaternions, _ISO_CODEBOOKS,
    )
    mx.random.seed(0)
    B, H, T, D = 1, 2, 50, 128
    n_groups = D // 4
    q_L = make_random_quaternions(n_groups, seed=7)
    q_R = make_random_quaternions(n_groups, seed=8) if mode == "full" else None
    centroids = _ISO_CODEBOOKS[bits]

    # Build a synthetic K cache.
    K = mx.random.normal((B * H * T, D)).astype(mx.float32)
    k_packed_flat, k_norms_flat = iso_compress(K, bits, q_L, q_R, centroids)
    k_packed = k_packed_flat.reshape(B, H, T, -1)
    k_norms = k_norms_flat.reshape(B, H, T)

    # Reference: decompress, then matmul against Q.
    K_dec = iso_decompress(k_packed_flat, k_norms_flat, D, bits, q_L, q_R, centroids)
    K_dec = K_dec.reshape(B, H, T, D)
    q = mx.random.normal((B, H, 1, D)).astype(mx.float32)
    scale = 1.0 / math.sqrt(D)
    ref = (q @ K_dec.swapaxes(-1, -2)) * scale

    # Fused kernel.
    fused = iso_fused_qk_scores(
        q, k_packed, k_norms, centroids, q_L, q_R, scale, D, bits,
    )
    mx.eval(ref, fused)
    diff = mx.max(mx.abs(ref - fused)).item()
    # Loose tolerance because the kernel runs everything in float32 (matching
    # ref) but rounds via shared-mem load/stores.
    assert diff < 5e-4, f"fused vs ref diff={diff:.3e}  (mode={mode}, bits={bits})"


@pytest.mark.parametrize("mode", ["full", "fast"])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_iso_fused_sv_matches_reference(mode, bits):
    """Fused SV value sum should match `iso_decompress + (probs @ V)` reference."""
    from turboquant.mlx_fused_iso_attention import (
        iso_compress, iso_decompress, iso_fused_sv_values,
        make_random_quaternions, _ISO_CODEBOOKS,
    )
    mx.random.seed(0)
    B, H, T, D = 1, 2, 40, 128
    n_groups = D // 4
    q_L = make_random_quaternions(n_groups, seed=21)
    q_R = make_random_quaternions(n_groups, seed=22) if mode == "full" else None
    centroids = _ISO_CODEBOOKS[bits]

    V = mx.random.normal((B * H * T, D)).astype(mx.float32)
    v_packed_flat, v_norms_flat = iso_compress(V, bits, q_L, q_R, centroids)
    v_packed = v_packed_flat.reshape(B, H, T, -1)
    v_norms = v_norms_flat.reshape(B, H, T)

    # softmax-style probability vector (random but normalized per head).
    raw = mx.random.uniform(shape=(B, H, 1, T))
    probs = raw / mx.sum(raw, axis=-1, keepdims=True)

    # Reference: decompress V, do matmul against probs.
    V_dec = iso_decompress(v_packed_flat, v_norms_flat, D, bits, q_L, q_R, centroids)
    V_dec = V_dec.reshape(B, H, T, D)
    ref = probs @ V_dec  # (B, H, 1, D)

    fused = iso_fused_sv_values(probs, v_packed, v_norms, centroids, q_L, q_R, D, bits)
    mx.eval(ref, fused)
    diff = mx.max(mx.abs(ref - fused)).item()
    assert diff < 5e-4, f"fused SV vs ref diff={diff:.3e}  (mode={mode}, bits={bits})"


# ── Flash decode (QK + softmax + SV in one kernel) ─────────────────────────


@pytest.mark.parametrize("mode", ["full", "fast"])
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("T", [64, 300])  # 300 exercises multi-tile + ragged last
def test_iso_flash_decode_matches_reference(mode, bits, T):
    """Fused flash decode == softmax(QK·scale) · V on the decompressed cache."""
    import math
    from turboquant.mlx_fused_iso_attention import (
        iso_compress, iso_decompress, iso_flash_decode,
        make_random_quaternions, _ISO_CODEBOOKS,
    )
    mx.random.seed(0)
    B, H, D = 1, 2, 128
    n_groups = D // 4
    q_L = make_random_quaternions(n_groups, seed=31)
    q_R = make_random_quaternions(n_groups, seed=32) if mode == "full" else None
    centroids = _ISO_CODEBOOKS[bits]

    K = mx.random.normal((B * H * T, D)).astype(mx.float32)
    V = mx.random.normal((B * H * T, D)).astype(mx.float32)
    k_packed_flat, k_norms_flat = iso_compress(K, bits, q_L, q_R, centroids)
    v_packed_flat, v_norms_flat = iso_compress(V, bits, q_L, q_R, centroids)
    k_packed = k_packed_flat.reshape(B, H, T, -1)
    k_norms = k_norms_flat.reshape(B, H, T)
    v_packed = v_packed_flat.reshape(B, H, T, -1)
    v_norms = v_norms_flat.reshape(B, H, T)

    q = mx.random.normal((B, H, 1, D)).astype(mx.float32)
    scale = 1.0 / math.sqrt(D)

    # Reference path: decompress, scaled-dot-product attention, no tricks.
    K_dec = iso_decompress(k_packed_flat, k_norms_flat, D, bits, q_L, q_R, centroids)
    V_dec = iso_decompress(v_packed_flat, v_norms_flat, D, bits, q_L, q_R, centroids)
    K_dec = K_dec.reshape(B, H, T, D)
    V_dec = V_dec.reshape(B, H, T, D)
    scores = (q @ K_dec.swapaxes(-1, -2)) * scale  # (B, H, 1, T)
    probs = mx.softmax(scores, axis=-1)
    ref = probs @ V_dec  # (B, H, 1, D)

    fused = iso_flash_decode(
        q, k_packed, k_norms, v_packed, v_norms,
        centroids, q_L, q_R, scale, D, bits,
    )
    mx.eval(ref, fused)
    diff = mx.max(mx.abs(ref - fused)).item()
    # Looser tolerance — flash decode runs an online softmax across tiles
    # which accumulates a bit more rounding than a one-shot softmax.
    assert diff < 5e-3, f"flash decode vs ref diff={diff:.3e}  (mode={mode}, bits={bits}, T={T})"
