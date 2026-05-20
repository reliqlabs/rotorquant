"""
MLX IsoQuant: 4D quaternion sandwich rotation + Lloyd-Max KV cache quantization.

Sibling of mlx_fused_planar_attention.py — same packing format and per-token
norm convention, but uses 4D quaternion-sandwich rotations instead of 2D
Givens. Quality matches the published IsoQuant numbers (PPL 6.91 vs PlanarQuant
7.05 on Llama 3.1 8B Q4_K_M / WikiText-2, both at 10.3x compression).

This module covers steps A–D of the MLX RotorQuant work:
    A. _planar/iso shared helpers (rotate, _compress, _decompress, codebooks)
    B. quaternion primitives (_quat_mul, _quat_conj, iso_rotate/unrotate)
    C. pure-MLX compress/decompress pipeline (no custom Metal yet)
    D. parity vs the torch reference (`turboquant.isoquant.IsoQuantMSE`)

The fused Metal kernels are intentionally NOT here — they're step E, a separate
follow-up. For inference you can compose `iso_compress` + a regular `mx.matmul`
attention path and still get the memory savings; you just won't see the
~1.99x decode speedup PR #8 reports for Planar until fused kernels land.

Conventions (must match the Metal kernels in mlx_fused_planar_attention.py):
    * packed (uint32): values_per_word = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    * norms_stored = original ||x|| (no further scaling) — kernel pattern is
      `centroid[idx] * norms[...]` to reconstruct in rotated space, then unrotate
    * codebooks: Lloyd-Max optimal for the actual coordinate distribution after
      a random orthogonal rotation, computed via `turboquant.lloyd_max.solve_lloyd_max`
      at module import for d=128 (Leanstral head_dim). Re-derive via
      `compute_codebooks(d, bits)` for other dims.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


# ── Quaternion primitives ────────────────────────────────────────────────────


def _quat_conj(q: mx.array) -> mx.array:
    """Quaternion conjugate: (w, x, y, z) -> (w, -x, -y, -z)."""
    signs = mx.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)
    return q * signs


def _quat_mul(a: mx.array, b: mx.array) -> mx.array:
    """Hamilton product. a, b: (..., 4) as (w, x, y, z). 16 muls + 12 adds."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return mx.stack([rw, rx, ry, rz], axis=-1)


def make_random_quaternions(n_groups: int, seed: int = 42, dtype=mx.float32) -> mx.array:
    """Random unit quaternions (n_groups, 4). Reproducible via seed.

    Uses an explicit `mx.random.key` so the global RNG state is untouched.
    """
    key = mx.random.key(seed)
    q = mx.random.normal(shape=(n_groups, 4), dtype=dtype, key=key)
    return q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


# ── Forward / inverse IsoQuant rotations ─────────────────────────────────────


def iso_rotate(x: mx.array, q_L: mx.array, q_R: Optional[mx.array] = None) -> mx.array:
    """
    Apply forward IsoQuant rotation on the last axis (treated as 4D blocks).

    Args:
        x: (..., d), d must be divisible by 4
        q_L: (n_groups, 4) unit quaternion per 4D block, n_groups == d / 4
        q_R: optional (n_groups, 4); if given, applies the full SO(4) sandwich
             T(v) = q_L * v * conj(q_R). If None, applies fast mode T(v) = q_L * v.

    Returns:
        (..., d) rotated tensor.
    """
    d = x.shape[-1]
    n_groups = q_L.shape[0]
    assert d == n_groups * 4, f"d={d} must equal n_groups*4 ({n_groups * 4})"

    blocks = x.reshape(*x.shape[:-1], n_groups, 4)
    if q_R is None:
        rotated = _quat_mul(q_L, blocks)
    else:
        rotated = _quat_mul(_quat_mul(q_L, blocks), _quat_conj(q_R))
    return rotated.reshape(*x.shape)


def iso_unrotate(x: mx.array, q_L: mx.array, q_R: Optional[mx.array] = None) -> mx.array:
    """
    Apply inverse IsoQuant rotation: T^{-1}(v) = conj(q_L) * v * q_R (full)
    or T^{-1}(v) = conj(q_L) * v (fast).

    This is the rotation that the CUDA fix on 2026-04-01 introduced — V dequant
    MUST use this inverse path, otherwise PPL explodes (15369 vs 7.05).
    """
    d = x.shape[-1]
    n_groups = q_L.shape[0]
    assert d == n_groups * 4

    blocks = x.reshape(*x.shape[:-1], n_groups, 4)
    if q_R is None:
        unrotated = _quat_mul(_quat_conj(q_L), blocks)
    else:
        unrotated = _quat_mul(_quat_mul(_quat_conj(q_L), blocks), q_R)
    return unrotated.reshape(*x.shape)


# ── Lloyd-Max codebooks ──────────────────────────────────────────────────────

# Default dim for codebook precomputation (Leanstral head_dim).
_DEFAULT_D = 128


def compute_codebooks(d: int, bits_list=(2, 3, 4)) -> dict:
    """Solve Lloyd-Max for each bit-width at the given vector dim.

    Returns a dict {bits: mx.array(2**bits) of float32 centroids}. Slow on first
    call (calls scipy.integrate.quad inside `solve_lloyd_max`); cache the result.
    """
    from .lloyd_max import solve_lloyd_max  # local import to avoid scipy at module load

    out = {}
    for bits in bits_list:
        centroids, _ = solve_lloyd_max(d, bits)
        out[bits] = mx.array(centroids.numpy(), dtype=mx.float32)
    return out


# Precomputed at import for the Leanstral head_dim.
_ISO_CODEBOOKS = compute_codebooks(_DEFAULT_D)


# ── Packing helpers (shared with PlanarQuant) ────────────────────────────────

_VALS_PER_WORD = {1: 32, 2: 16, 3: 10, 4: 8}


def _pack(indices: mx.array, bits: int) -> mx.array:
    """Pack last-axis quantization indices into uint32 words.

    Layout matches the Metal kernels: word holds vals_per_word values, value i
    occupies bits [i*bits, (i+1)*bits). Trailing bits in a word are unused (e.g.,
    2 bits at the top of a 3-bit word).
    """
    vpw = _VALS_PER_WORD[bits]
    d = indices.shape[-1]
    packed_dim = (d + vpw - 1) // vpw
    pad = packed_dim * vpw - d
    if pad:
        pad_widths = [(0, 0)] * (indices.ndim - 1) + [(0, pad)]
        indices = mx.pad(indices, pad_widths)
    grouped = indices.reshape(*indices.shape[:-1], packed_dim, vpw).astype(mx.uint32)
    shifts = (mx.arange(vpw, dtype=mx.uint32) * bits)
    return mx.sum(grouped << shifts, axis=-1)


def _unpack(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Inverse of `_pack`. Returns (..., dim) uint32 indices."""
    vpw = _VALS_PER_WORD[bits]
    bit_mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    shifts = (mx.arange(vpw, dtype=mx.uint32) * bits)
    # packed: (..., packed_dim)  -> (..., packed_dim, vpw)
    expanded = (packed[..., None] >> shifts) & bit_mask
    flat = expanded.reshape(*expanded.shape[:-2], -1)
    return flat[..., :dim]


# ── Pure-MLX compress / decompress ───────────────────────────────────────────


def iso_compress(
    x: mx.array,
    bits: int,
    q_L: mx.array,
    q_R: Optional[mx.array] = None,
    centroids: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """
    Compress a batch of vectors via IsoQuant: normalize -> rotate -> quantize -> pack.

    Args:
        x: (..., d) inputs
        bits: 1..4
        q_L, q_R: quaternions returned by `make_random_quaternions`
        centroids: optional codebook override (default: `_ISO_CODEBOOKS[bits]`)

    Returns:
        packed: (..., packed_dim) uint32
        norms:  (...,) float32 — original ||x|| (kernel multiplies centroid by this)
    """
    if centroids is None:
        centroids = _ISO_CODEBOOKS[bits]

    x_f = x.astype(mx.float32)
    norms = mx.linalg.norm(x_f, axis=-1, keepdims=True)
    x_unit = x_f / mx.maximum(norms, 1e-8)
    rotated = iso_rotate(x_unit, q_L, q_R)

    # Nearest-centroid quantization (per-coordinate)
    diffs = mx.abs(rotated[..., None] - centroids)
    indices = mx.argmin(diffs, axis=-1).astype(mx.uint32)
    packed = _pack(indices, bits)
    return packed, norms.squeeze(-1)


def iso_decompress(
    packed: mx.array,
    norms: mx.array,
    dim: int,
    bits: int,
    q_L: mx.array,
    q_R: Optional[mx.array] = None,
    centroids: Optional[mx.array] = None,
    dtype=mx.float32,
) -> mx.array:
    """Reverse of `iso_compress`. Returns (..., dim) reconstructed in original space."""
    if centroids is None:
        centroids = _ISO_CODEBOOKS[bits]

    indices = _unpack(packed, bits, dim).astype(mx.int32)
    values = centroids[indices]  # (..., dim) — in rotated unit space

    # Move to rotated full-scale via per-token norm: kernel pattern `centroid * norms`
    rotated_full = values * norms[..., None]
    return iso_unrotate(rotated_full, q_L, q_R).astype(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Step E (Metal fusion) — building block #1: inverse quaternion rotation
#
# This is the foundation for the fused QK / SV / flash-decode IsoQuant kernels
# that mirror PR #8's PlanarQuant family. Each of those kernels does a small
# inverse rotation in shared memory before the dot product or accumulation —
# the math below is exactly that step, lifted out as a standalone kernel so we
# can validate it independently before composing it into a full attention path.
# ─────────────────────────────────────────────────────────────────────────────


ISO_INVERSE_ROTATE_KERNEL = """
    uint row = threadgroup_position_in_grid.x;     // which row of (B, d)
    uint elem = thread_position_in_threadgroup.x;  // element index in [0, d)
    uint d = dims[0];
    uint n_groups = dims[1];
    uint has_qR = dims[2];

    // Stage row into shared memory so threads in a 4-group can read peers.
    threadgroup float v_shared[1024];
    v_shared[elem] = (float)x[row * d + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each 4-block is handled by the thread whose elem index is its starting position.
    if (elem % 4 == 0) {
        uint group_idx = elem / 4;
        uint qbase = group_idx * 4;

        float v0 = v_shared[elem];
        float v1 = v_shared[elem + 1];
        float v2 = v_shared[elem + 2];
        float v3 = v_shared[elem + 3];

        // conj(q_L) = (qlw, -qlx, -qly, -qlz)
        float qlw =  q_L[qbase + 0];
        float qlx = -q_L[qbase + 1];
        float qly = -q_L[qbase + 2];
        float qlz = -q_L[qbase + 3];

        // temp = conj(q_L) * v  (Hamilton product)
        float tw = qlw * v0 - qlx * v1 - qly * v2 - qlz * v3;
        float tx = qlw * v1 + qlx * v0 + qly * v3 - qlz * v2;
        float ty = qlw * v2 - qlx * v3 + qly * v0 + qlz * v1;
        float tz = qlw * v3 + qlx * v2 - qly * v1 + qlz * v0;

        float rw, rx, ry, rz;
        if (has_qR == 1u) {
            float qrw = q_R[qbase + 0];
            float qrx = q_R[qbase + 1];
            float qry = q_R[qbase + 2];
            float qrz = q_R[qbase + 3];
            rw = tw * qrw - tx * qrx - ty * qry - tz * qrz;
            rx = tw * qrx + tx * qrw + ty * qrz - tz * qry;
            ry = tw * qry - tx * qrz + ty * qrw + tz * qrx;
            rz = tw * qrz + tx * qry - ty * qrx + tz * qrw;
        } else {
            rw = tw; rx = tx; ry = ty; rz = tz;
        }

        v_shared[elem]     = rw;
        v_shared[elem + 1] = rx;
        v_shared[elem + 2] = ry;
        v_shared[elem + 3] = rz;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[row * d + elem] = (T)v_shared[elem];
"""


_iso_inverse_rotate_kernel = None


ISO_FUSED_QK_KERNEL = """
    uint seq_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint has_qR = dims[6];
    uint bit_mask = (1u << bits) - 1u;

    // Load Q into shared memory.
    threadgroup float q_shared[1024];
    q_shared[elem] = (float)query[head_idx * dim + elem];

    // Unpack K element from packed uint32.
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    // Codebook lookup × per-token norm.
    float val = centroids[idx] * norms[head_idx * seq_len + seq_idx];

    threadgroup float k_shared[1024];
    k_shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse quaternion sandwich: one thread per 4-group.
    if (elem % 4 == 0) {
        uint group_idx = elem / 4;
        uint qbase = group_idx * 4;

        float v0 = k_shared[elem];
        float v1 = k_shared[elem + 1];
        float v2 = k_shared[elem + 2];
        float v3 = k_shared[elem + 3];

        float qlw =  q_L[qbase + 0];
        float qlx = -q_L[qbase + 1];
        float qly = -q_L[qbase + 2];
        float qlz = -q_L[qbase + 3];

        float tw = qlw * v0 - qlx * v1 - qly * v2 - qlz * v3;
        float tx = qlw * v1 + qlx * v0 + qly * v3 - qlz * v2;
        float ty = qlw * v2 - qlx * v3 + qly * v0 + qlz * v1;
        float tz = qlw * v3 + qlx * v2 - qly * v1 + qlz * v0;

        if (has_qR == 1u) {
            float qrw = q_R[qbase + 0];
            float qrx = q_R[qbase + 1];
            float qry = q_R[qbase + 2];
            float qrz = q_R[qbase + 3];
            k_shared[elem]     = tw * qrw - tx * qrx - ty * qry - tz * qrz;
            k_shared[elem + 1] = tw * qrx + tx * qrw + ty * qrz - tz * qry;
            k_shared[elem + 2] = tw * qry - tx * qrz + ty * qrw + tz * qrx;
            k_shared[elem + 3] = tw * qrz + tx * qry - ty * qrx + tz * qrw;
        } else {
            k_shared[elem]     = tw;
            k_shared[elem + 1] = tx;
            k_shared[elem + 2] = ty;
            k_shared[elem + 3] = tz;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dot product Q · K_inv-rotated.
    float dot = q_shared[elem] * k_shared[elem];
    threadgroup float dot_shared[1024];
    dot_shared[elem] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            dot_shared[elem] += dot_shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (elem == 0) {
        out[head_idx * seq_len + seq_idx] = (T)(dot_shared[0] * scale[0]);
    }
"""


_iso_fused_qk_kernel = None


def iso_fused_qk_scores(
    query: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    q_L: mx.array,
    q_R: Optional[mx.array],
    scale: float,
    dim: int,
    bits: int,
) -> mx.array:
    """One-pass QK score: unpack K → inverse iso-rotate → dot with Q → scale.

    Mirrors `planar_fused_qk_scores` in mlx_fused_planar_attention.py, but
    uses the 4D quaternion sandwich rotation instead of the 2D Givens. Same
    packed format and per-token norm convention so existing K cache buffers
    are reusable.

    Args:
        query: (B, H, 1, D) float / half — the query for the current decode step.
        k_packed: (B, H, T, packed_dim) uint32 — packed iso-quantized K cache.
        k_norms:  (B, H, T) float32 — per-token K norm (matches kernel
            convention `centroid[idx] * norm`).
        centroids: (n_levels,) float32 — Lloyd-Max codebook for `bits`.
        q_L: (n_groups, 4) float32 — left quaternion rotor per 4-block.
        q_R: (n_groups, 4) float32 or None — right rotor (None = SO(3) fast mode).
        scale: softmax temperature, typically 1/sqrt(D).
        dim: head_dim (must be divisible by 4 and ≤ 1024).
        bits: quantization bits (1..4).

    Returns:
        scores (B, H, 1, T) float32 — QK scaled scores.
    """
    global _iso_fused_qk_kernel
    if _iso_fused_qk_kernel is None:
        _iso_fused_qk_kernel = mx.fast.metal_kernel(
            name="iso_fused_qk",
            input_names=[
                "query", "packed", "norms", "centroids",
                "q_L", "q_R", "scale", "dims",
            ],
            output_names=["out"],
            source=ISO_FUSED_QK_KERNEL,
        )

    B = query.shape[0]
    H = query.shape[1]
    seq_len = k_norms.shape[2]
    p_dim = k_packed.shape[-1]
    vpw = _VALS_PER_WORD[bits]
    assert dim % 4 == 0 and dim <= 1024, f"dim={dim} unsupported"
    assert q_L.shape[0] * 4 == dim, f"q_L groups must = dim/4"

    has_qR = 1 if q_R is not None else 0
    q_R_flat = (q_R if q_R is not None else mx.zeros_like(q_L)).astype(mx.float32).reshape(-1)
    q_L_flat = q_L.astype(mx.float32).reshape(-1)

    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array(
        [dim, seq_len, B * H, bits, vpw, p_dim, has_qR],
        dtype=mx.uint32,
    )

    outputs = _iso_fused_qk_kernel(
        inputs=[
            query.astype(mx.float32).reshape(B * H * dim),
            k_packed.astype(mx.uint32).reshape(B * H * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(B * H * seq_len),
            centroids, q_L_flat, q_R_flat, scale_arr, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, B * H, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(B * H * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(B, H, 1, seq_len)


def iso_unrotate_metal(x: mx.array, q_L: mx.array, q_R: Optional[mx.array] = None) -> mx.array:
    """Metal-fused inverse IsoQuant rotation. Equivalent to `iso_unrotate` but
    runs a single Metal threadgroup per row.

    x: (..., d) — flattens the leading dims into row index for the kernel.
    q_L: (n_groups, 4); q_R: (n_groups, 4) or None for fast mode.
    Returns: same shape as `x`.
    """
    global _iso_inverse_rotate_kernel
    if _iso_inverse_rotate_kernel is None:
        _iso_inverse_rotate_kernel = mx.fast.metal_kernel(
            name="iso_inverse_rotate",
            input_names=["x", "q_L", "q_R", "dims"],
            output_names=["out"],
            source=ISO_INVERSE_ROTATE_KERNEL,
        )

    d = x.shape[-1]
    n_groups = q_L.shape[0]
    assert d == n_groups * 4, f"d={d} must equal n_groups*4 ({n_groups * 4})"
    assert d <= 1024, "ISO_INVERSE_ROTATE_KERNEL shared-memory layout assumes d ≤ 1024"

    original_shape = x.shape
    n_rows = 1
    for s in original_shape[:-1]:
        n_rows *= s

    x_flat = x.astype(mx.float32).reshape(n_rows, d)
    q_L_flat = q_L.astype(mx.float32).reshape(-1)  # (n_groups * 4,)
    has_qR = 1 if q_R is not None else 0
    q_R_flat = (q_R if q_R is not None else mx.zeros_like(q_L)).astype(mx.float32).reshape(-1)
    dims_arr = mx.array([d, n_groups, has_qR], dtype=mx.uint32)

    outputs = _iso_inverse_rotate_kernel(
        inputs=[x_flat, q_L_flat, q_R_flat, dims_arr],
        template=[("T", mx.float32)],
        grid=(n_rows * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(n_rows * d,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(original_shape).astype(x.dtype)
