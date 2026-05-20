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
        q_L: (n_groups, 4) for a single rotor broadcast across leading dims, OR
             (H, n_groups, 4) for per-head rotors — in which case x must have
             shape (..., H, d) so the head axis at -2 aligns with q_L's first
             axis under MLX broadcasting.
        q_R: optional, same shape as q_L. None => SO(3) fast mode T(v) = q_L * v.

    Returns:
        (..., d) rotated tensor.
    """
    d = x.shape[-1]
    if q_L.ndim == 2:
        n_groups = q_L.shape[0]
    else:
        n_groups = q_L.shape[1]
        assert x.shape[-2] == q_L.shape[0], (
            f"per-head iso_rotate: x must have heads at axis -2 matching "
            f"q_L.shape[0]={q_L.shape[0]}, got x.shape={x.shape}"
        )
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
    or T^{-1}(v) = conj(q_L) * v (fast). Same broadcasting rules as iso_rotate.
    """
    d = x.shape[-1]
    if q_L.ndim == 2:
        n_groups = q_L.shape[0]
    else:
        n_groups = q_L.shape[1]
        assert x.shape[-2] == q_L.shape[0], (
            f"per-head iso_unrotate: x must have heads at axis -2 matching "
            f"q_L.shape[0]={q_L.shape[0]}, got x.shape={x.shape}"
        )
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


def _prepare_rotors_for_kernel(
    q_L: mx.array, q_R: Optional[mx.array], dim: int
) -> Tuple[mx.array, mx.array, int]:
    """Normalize q_L / q_R into a flat buffer + n_heads_real for kernel use.

    Accepts either:
      - 2D q_L of shape (n_groups, 4): legacy per-layer rotor, broadcast across
        every head by the kernel. Returns n_heads_real=1.
      - 3D q_L of shape (n_heads, n_groups, 4): per-head rotors. Returns
        n_heads_real = q_L.shape[0]; kernel indexes by head.

    Always returns q_R as a float32 buffer of the same shape as q_L (zeros
    when q_R is None — the kernel ignores it via has_qR=0). This lets the
    kernel signature be uniform.
    """
    if q_L.ndim == 2:
        n_heads_real = 1
        assert q_L.shape[0] * 4 == dim, (
            f"q_L shape {tuple(q_L.shape)} incompatible with dim={dim}"
        )
    elif q_L.ndim == 3:
        n_heads_real = q_L.shape[0]
        assert q_L.shape[1] * 4 == dim, (
            f"q_L per-head shape {tuple(q_L.shape)} incompatible with dim={dim}"
        )
    else:
        raise ValueError(f"q_L must be 2D or 3D, got ndim={q_L.ndim}")

    q_L_flat = q_L.astype(mx.float32).reshape(-1)
    if q_R is not None:
        assert q_R.shape == q_L.shape, (
            f"q_L shape {tuple(q_L.shape)} != q_R shape {tuple(q_R.shape)}"
        )
        q_R_flat = q_R.astype(mx.float32).reshape(-1)
    else:
        q_R_flat = mx.zeros_like(q_L_flat)
    return q_L_flat, q_R_flat, n_heads_real


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
    uint n_heads_real = dims[3];     // 1 = broadcast across heads (legacy)
    uint rows_per_head = dims[4];    // ignored when n_heads_real == 1
    uint head_in_layer = (n_heads_real <= 1u) ? 0u
                        : ((row / rows_per_head) % n_heads_real);

    // Stage row into shared memory so threads in a 4-group can read peers.
    threadgroup float v_shared[1024];
    v_shared[elem] = (float)x[row * d + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each 4-block is handled by the thread whose elem index is its starting position.
    if (elem % 4 == 0) {
        uint group_idx = elem / 4;
        uint qbase = head_in_layer * d + group_idx * 4;

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
    uint n_heads_real = dims[7];   // 1 = broadcast (legacy); else H
    uint bit_mask = (1u << bits) - 1u;
    uint head_in_layer = head_idx % n_heads_real;

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
        uint qbase = head_in_layer * dim + group_idx * 4;

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


ISO_FUSED_SV_KERNEL = """
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint has_qR = dims[6];
    uint n_heads_real = dims[7];
    uint bit_mask = (1u << bits) - 1u;
    uint head_in_layer = head_idx % n_heads_real;

    float acc = 0.0f;
    threadgroup float v_shared[1024];

    for (uint seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        // Unpack V[seq_idx][elem] and apply per-token norm.
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        float val = centroids[idx] * norms[head_idx * seq_len + seq_idx];

        v_shared[elem] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inverse quaternion sandwich on each 4-group (per-token).
        if (elem % 4 == 0) {
            uint group_idx = elem / 4;
            uint qbase = head_in_layer * dim + group_idx * 4;

            float v0 = v_shared[elem];
            float v1 = v_shared[elem + 1];
            float v2 = v_shared[elem + 2];
            float v3 = v_shared[elem + 3];

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
                v_shared[elem]     = tw * qrw - tx * qrx - ty * qry - tz * qrz;
                v_shared[elem + 1] = tw * qrx + tx * qrw + ty * qrz - tz * qry;
                v_shared[elem + 2] = tw * qry - tx * qrz + ty * qrw + tz * qrx;
                v_shared[elem + 3] = tw * qrz + tx * qry - ty * qrx + tz * qrw;
            } else {
                v_shared[elem]     = tw;
                v_shared[elem + 1] = tx;
                v_shared[elem + 2] = ty;
                v_shared[elem + 3] = tz;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate softmax-weighted V into per-thread (per-elem) acc.
        float prob = (float)probs[head_idx * seq_len + seq_idx];
        acc += prob * v_shared[elem];
    }

    out[head_idx * dim + elem] = (T)acc;
"""


_iso_fused_sv_kernel = None


def iso_fused_sv_values(
    probs: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    q_L: mx.array,
    q_R: Optional[mx.array],
    dim: int,
    bits: int,
) -> mx.array:
    """Weighted V sum: sum over T of (prob[t] · iso-unrotated V[t]).

    Mirrors `planar_fused_sv_values` in mlx_fused_planar_attention.py but uses
    the quaternion sandwich rotation. Same caveat about ≤ 1024-thread shared
    memory budget — for d > 256 use the tiled variant (TBD as a follow-up).

    Args:
        probs: (B, H, 1, T) float — softmax probabilities from the QK step.
        v_packed: (B, H, T, packed_dim) uint32 — packed iso-quantized V cache.
        v_norms:  (B, H, T) float32 — per-token V norm.
        centroids, q_L, q_R, dim, bits: as for `iso_fused_qk_scores`.

    Returns:
        out (B, H, 1, dim) float32 — attention-weighted V.
    """
    global _iso_fused_sv_kernel
    if _iso_fused_sv_kernel is None:
        _iso_fused_sv_kernel = mx.fast.metal_kernel(
            name="iso_fused_sv",
            input_names=[
                "probs", "packed", "norms", "centroids",
                "q_L", "q_R", "dims",
            ],
            output_names=["out"],
            source=ISO_FUSED_SV_KERNEL,
        )

    B = probs.shape[0]
    H = probs.shape[1]
    seq_len = v_norms.shape[2]
    p_dim = v_packed.shape[-1]
    vpw = _VALS_PER_WORD[bits]
    assert dim % 4 == 0 and dim <= 1024

    has_qR = 1 if q_R is not None else 0
    q_L_flat, q_R_flat, n_heads_real = _prepare_rotors_for_kernel(q_L, q_R, dim)
    dims_arr = mx.array(
        [dim, seq_len, B * H, bits, vpw, p_dim, has_qR, n_heads_real],
        dtype=mx.uint32,
    )

    outputs = _iso_fused_sv_kernel(
        inputs=[
            probs.astype(mx.float32).reshape(B * H * seq_len),
            v_packed.astype(mx.uint32).reshape(B * H * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(B * H * seq_len),
            centroids, q_L_flat, q_R_flat, dims_arr,
        ],
        template=[("T", mx.float32)],
        # 1 threadgroup per head; dim threads per threadgroup.
        grid=(dim, B * H, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(B * H * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(B, H, 1, dim)


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

    has_qR = 1 if q_R is not None else 0
    q_L_flat, q_R_flat, n_heads_real = _prepare_rotors_for_kernel(q_L, q_R, dim)

    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array(
        [dim, seq_len, B * H, bits, vpw, p_dim, has_qR, n_heads_real],
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


# ── Fully fused flash decode: QK + online softmax + SV in one kernel ────────
#
# Per-tile kernel mirrors `PLANAR_FLASH_DECODE_KERNEL` in the planar module.
# Reads packed K and V exactly once, never materializes the decompressed
# tensors in device memory. Output is partial-per-tile + log-sum-exp metadata;
# the Python wrapper merges tiles into the final attention output.


ISO_TILE_SIZE = 256


ISO_FLASH_DECODE_KERNEL = """
    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint has_qR = dims[7];
    uint n_heads_real = dims[8];
    uint bit_mask = (1u << bits) - 1u;
    uint head_in_layer = head_idx % n_heads_real;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    // Load Q once for this tile.
    threadgroup float q_shared[1024];
    q_shared[elem] = (float)query[head_idx * dim + elem];

    // Online softmax shared singletons (broadcast from thread 0).
    threadgroup float s_corr[1];
    threadgroup float s_expsc[1];
    threadgroup float s_max[1];
    threadgroup float s_sum[1];
    if (elem == 0) {
        s_max[0] = -1e30f;
        s_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float kv_shared[1024];
    threadgroup float dot_shared[1024];
    float acc_v = 0.0f;

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        // ── Unpack K element + apply norm ────────────────────────────────
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint k_word = k_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint k_idx = (k_word >> (pos_in_word * bits)) & bit_mask;
        float k_val = centroids[k_idx] * k_norms[head_idx * seq_len + seq_idx];

        kv_shared[elem] = k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Inverse quaternion sandwich on K (4-block per thread%4==0) ───
        if (elem % 4 == 0) {
            uint group_idx = elem / 4;
            uint qbase = head_in_layer * dim + group_idx * 4;

            float v0 = kv_shared[elem];
            float v1 = kv_shared[elem + 1];
            float v2 = kv_shared[elem + 2];
            float v3 = kv_shared[elem + 3];

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
                kv_shared[elem]     = tw * qrw - tx * qrx - ty * qry - tz * qrz;
                kv_shared[elem + 1] = tw * qrx + tx * qrw + ty * qrz - tz * qry;
                kv_shared[elem + 2] = tw * qry - tx * qrz + ty * qrw + tz * qrx;
                kv_shared[elem + 3] = tw * qrz + tx * qry - ty * qrx + tz * qrw;
            } else {
                kv_shared[elem]     = tw;
                kv_shared[elem + 1] = tx;
                kv_shared[elem + 2] = ty;
                kv_shared[elem + 3] = tz;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── QK dot product + tree reduce ─────────────────────────────────
        dot_shared[elem] = q_shared[elem] * kv_shared[elem];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = dim / 2; stride > 0; stride >>= 1) {
            if (elem < stride) dot_shared[elem] += dot_shared[elem + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Online softmax update (thread 0 broadcasts) ──────────────────
        if (elem == 0) {
            float score = dot_shared[0] * scale[0];
            float old_max = s_max[0];
            float new_max = (score > old_max) ? score : old_max;
            float corr = exp(old_max - new_max);
            float es = exp(score - new_max);
            s_max[0] = new_max;
            s_sum[0] = s_sum[0] * corr + es;
            s_corr[0] = corr;
            s_expsc[0] = es;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = s_corr[0];
        float es = s_expsc[0];

        // ── Rescale accumulated V by softmax correction ──────────────────
        acc_v = acc_v * corr;

        // ── Unpack V element + apply norm ────────────────────────────────
        uint v_word = v_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint v_idx = (v_word >> (pos_in_word * bits)) & bit_mask;
        float v_val = centroids[v_idx] * v_norms[head_idx * seq_len + seq_idx];

        kv_shared[elem] = v_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Inverse quaternion sandwich on V ─────────────────────────────
        if (elem % 4 == 0) {
            uint group_idx = elem / 4;
            uint qbase = head_in_layer * dim + group_idx * 4;

            float v0 = kv_shared[elem];
            float v1 = kv_shared[elem + 1];
            float v2 = kv_shared[elem + 2];
            float v3 = kv_shared[elem + 3];

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
                kv_shared[elem]     = tw * qrw - tx * qrx - ty * qry - tz * qrz;
                kv_shared[elem + 1] = tw * qrx + tx * qrw + ty * qrz - tz * qry;
                kv_shared[elem + 2] = tw * qry - tx * qrz + ty * qrw + tz * qrx;
                kv_shared[elem + 3] = tw * qrz + tx * qry - ty * qrx + tz * qrw;
            } else {
                kv_shared[elem]     = tw;
                kv_shared[elem + 1] = tx;
                kv_shared[elem + 2] = ty;
                kv_shared[elem + 3] = tz;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Accumulate softmax-weighted V ────────────────────────────────
        acc_v += es * kv_shared[elem];
    }

    // Write partial output (unnormalized) + per-tile log-sum-exp metadata.
    uint out_base = (tile_idx * n_heads + head_idx) * dim;
    partial_o[out_base + elem] = acc_v;

    if (elem == 0) {
        uint meta_idx = tile_idx * n_heads + head_idx;
        tile_max[meta_idx] = s_max[0];
        tile_sum_exp[meta_idx] = s_sum[0];
    }
"""


_iso_flash_decode_kernel = None


def iso_flash_decode(
    query: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    q_L: mx.array,
    q_R: Optional[mx.array],
    scale: float,
    dim: int,
    bits: int,
) -> mx.array:
    """Fully fused IsoQuant flash decode.

    Each threadgroup processes a `ISO_TILE_SIZE`-token tile for one head, doing
    unpack(K) → inverse-iso-rotate(K) → QK dot → online softmax update →
    unpack(V) → inverse-iso-rotate(V) → accumulate(es * V) without writing
    any FP intermediate K/V to device memory. Tiles are merged via log-sum-exp
    in Python on the way out.

    Args:
        query: (B, H, 1, D) — current decode-step query.
        k_packed, k_norms: iso-packed K cache (see iso_compress for layout).
        v_packed, v_norms: iso-packed V cache.
        centroids, q_L, q_R, scale, dim, bits: as for `iso_fused_qk_scores`.

    Returns:
        out (B, H, 1, D) — attention output. Equivalent to a softmax(QK·scale)·V
        over the full T-length context, but reads packed K/V exactly once.
    """
    global _iso_flash_decode_kernel
    if _iso_flash_decode_kernel is None:
        _iso_flash_decode_kernel = mx.fast.metal_kernel(
            name="iso_flash_decode",
            input_names=[
                "query", "k_packed", "k_norms", "v_packed", "v_norms",
                "centroids", "q_L", "q_R", "scale", "dims",
            ],
            output_names=["partial_o", "tile_max", "tile_sum_exp"],
            source=ISO_FLASH_DECODE_KERNEL,
        )

    B = query.shape[0]
    H = query.shape[1]
    seq_len = k_norms.shape[2]
    p_dim = k_packed.shape[-1]
    vpw = _VALS_PER_WORD[bits]
    num_tiles = (seq_len + ISO_TILE_SIZE - 1) // ISO_TILE_SIZE
    n_bh = B * H
    assert dim % 4 == 0 and dim <= 1024

    has_qR = 1 if q_R is not None else 0
    q_L_flat, q_R_flat, n_heads_real = _prepare_rotors_for_kernel(q_L, q_R, dim)

    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array(
        [dim, seq_len, n_bh, bits, vpw, p_dim, ISO_TILE_SIZE, has_qR, n_heads_real],
        dtype=mx.uint32,
    )

    outputs = _iso_flash_decode_kernel(
        inputs=[
            query.astype(mx.float32).reshape(n_bh * dim),
            k_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_bh * seq_len),
            v_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(n_bh * seq_len),
            centroids, q_L_flat, q_R_flat, scale_arr, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, n_bh, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[
            (num_tiles * n_bh * dim,),
            (num_tiles * n_bh,),
            (num_tiles * n_bh,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    partial_o = outputs[0].reshape(num_tiles, n_bh, dim)
    t_max = outputs[1].reshape(num_tiles, n_bh, 1)
    t_sum_exp = outputs[2].reshape(num_tiles, n_bh, 1)

    # Exact log-sum-exp merge across tiles.
    global_max = mx.max(t_max, axis=0, keepdims=True)
    corrections = mx.exp(t_max - global_max)
    numerator = mx.sum(partial_o * corrections, axis=0)
    denominator = mx.sum(t_sum_exp * corrections, axis=0)
    result = numerator / (denominator + 1e-8)

    return result.reshape(B, H, 1, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Two-pass sparse attention (mirrors planar PR #8's phase1/phase2 design).
#
# At long context, dense flash decode dominated by V-decompression work for
# tokens whose attention weight is ~0 after softmax. The two-pass split:
#   phase1: score ALL tokens (cheap QK only) + emit per-tile top scores
#   bridge: pick the topk-th highest score per head as a threshold
#   phase2: do online-softmax + V-decompress + accumulate ONLY for tokens
#           whose score >= threshold. Whole tiles with no survivors skip
#           the V decompress entirely (huge win at sparse top-k).
#
# Approximation: phase1 thresholds by score (pre-softmax), so we keep the
# top-k *attention logits*. This is the same approximation FlashAttention-Sparse
# and PR #8 use; it's exact when top-k captures all non-negligible softmax mass.
# ─────────────────────────────────────────────────────────────────────────────


ISO_PHASE1_SCORE_KERNEL = """
    // Phase 1: For each (tile, head), score every token in the tile (QK
    // with iso decompress on K), write per-token score to all_scores, and
    // track the top-4 scores in this tile for threshold derivation.
    //
    // grid:        (num_tiles * dim, n_bh, 1)
    // threadgroup: (dim, 1, 1)

    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint has_qR = dims[7];
    uint n_heads_real = dims[8];
    uint bit_mask = (1u << bits) - 1u;
    uint head_in_layer = head_idx % n_heads_real;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    // Load Q once.
    threadgroup float q_shared[1024];
    q_shared[elem] = (float)query[head_idx * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float k_shared[1024];
    threadgroup float dot_shared[1024];

    // Top-4 in this tile (matches planar's top-4-per-tile budget).
    threadgroup float tile_tops[4];
    if (elem < 4) tile_tops[elem] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        // ── Unpack K element + apply per-token norm ──────────────────────
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint k_word = k_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint k_idx = (k_word >> (pos_in_word * bits)) & bit_mask;
        float k_val = centroids[k_idx] * k_norms[head_idx * seq_len + seq_idx];

        k_shared[elem] = k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Inverse iso (quaternion sandwich) on K ───────────────────────
        if (elem % 4 == 0) {
            uint group_idx = elem / 4;
            uint qbase = head_in_layer * dim + group_idx * 4;

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

        // ── QK dot + tree reduce ────────────────────────────────────────
        dot_shared[elem] = q_shared[elem] * k_shared[elem];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = dim / 2; stride > 0; stride >>= 1) {
            if (elem < stride) dot_shared[elem] += dot_shared[elem + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (elem == 0) {
            float score = dot_shared[0] * scale[0];
            all_scores[head_idx * seq_len + seq_idx] = score;

            // Insertion sort into top-4 (tile_tops kept descending).
            if (score > tile_tops[3]) {
                tile_tops[3] = score;
                for (int i = 2; i >= 0; i--) {
                    if (tile_tops[i+1] > tile_tops[i]) {
                        float tmp = tile_tops[i];
                        tile_tops[i] = tile_tops[i+1];
                        tile_tops[i+1] = tmp;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (elem < 4) {
        uint base = (tile_idx * n_heads + head_idx) * 4;
        tile_top_scores[base + elem] = tile_tops[elem];
    }
"""


ISO_PHASE2_SPARSE_ATTEND_KERNEL = """
    // Phase 2: For each (tile, head), read precomputed all_scores; for
    // tokens with score >= threshold[head], do online softmax + V iso
    // decompress + accumulate. Tiles whose every token is below threshold
    // exit immediately (no V decompress, no barriers in the hot loop).
    //
    // Output is partial_o + tile_max + tile_sum_exp — same shape and
    // semantics as ISO_FLASH_DECODE_KERNEL so the LSE merge in Python is
    // unchanged.

    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint has_qR = dims[7];
    uint n_heads_real = dims[8];
    uint bit_mask = (1u << bits) - 1u;
    uint head_in_layer = head_idx % n_heads_real;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    float threshold_val = threshold[head_idx];

    // ── Tile-level early exit: skip entire tile if no survivors ──────
    threadgroup bool tile_has_survivors[1];
    if (elem == 0) {
        tile_has_survivors[0] = false;
        for (uint i = tile_start; i < tile_end; i++) {
            if (all_scores[head_idx * seq_len + i] >= threshold_val) {
                tile_has_survivors[0] = true;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!tile_has_survivors[0]) {
        uint out_base = (tile_idx * n_heads + head_idx) * dim;
        partial_o[out_base + elem] = 0.0f;
        if (elem == 0) {
            uint meta = tile_idx * n_heads + head_idx;
            tile_max[meta] = -1e30f;
            tile_sum_exp[meta] = 0.0f;
        }
        return;
    }

    // Online softmax state.
    threadgroup float s_max[1];
    threadgroup float s_sum[1];
    threadgroup float s_corr[1];
    threadgroup float s_expsc[1];
    if (elem == 0) {
        s_max[0] = -1e30f;
        s_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float v_shared[1024];
    float acc_v = 0.0f;

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        float score = all_scores[head_idx * seq_len + seq_idx];
        if (score < threshold_val) continue;

        // Online softmax update.
        if (elem == 0) {
            float old_max = s_max[0];
            float new_max = (score > old_max) ? score : old_max;
            float corr = exp(old_max - new_max);
            float es = exp(score - new_max);
            s_max[0] = new_max;
            s_sum[0] = s_sum[0] * corr + es;
            s_corr[0] = corr;
            s_expsc[0] = es;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = s_corr[0];
        float es = s_expsc[0];
        acc_v = acc_v * corr;

        // ── Unpack V element + apply norm ────────────────────────────────
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint v_word = v_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint v_idx = (v_word >> (pos_in_word * bits)) & bit_mask;
        float v_val = centroids[v_idx] * v_norms[head_idx * seq_len + seq_idx];

        v_shared[elem] = v_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Inverse iso on V ────────────────────────────────────────────
        if (elem % 4 == 0) {
            uint group_idx = elem / 4;
            uint qbase = head_in_layer * dim + group_idx * 4;

            float v0 = v_shared[elem];
            float v1 = v_shared[elem + 1];
            float v2 = v_shared[elem + 2];
            float v3 = v_shared[elem + 3];

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
                v_shared[elem]     = tw * qrw - tx * qrx - ty * qry - tz * qrz;
                v_shared[elem + 1] = tw * qrx + tx * qrw + ty * qrz - tz * qry;
                v_shared[elem + 2] = tw * qry - tx * qrz + ty * qrw + tz * qrx;
                v_shared[elem + 3] = tw * qrz + tx * qry - ty * qrx + tz * qrw;
            } else {
                v_shared[elem]     = tw;
                v_shared[elem + 1] = tx;
                v_shared[elem + 2] = ty;
                v_shared[elem + 3] = tz;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        acc_v += es * v_shared[elem];
    }

    uint out_base = (tile_idx * n_heads + head_idx) * dim;
    partial_o[out_base + elem] = acc_v;

    if (elem == 0) {
        uint meta = tile_idx * n_heads + head_idx;
        tile_max[meta] = s_max[0];
        tile_sum_exp[meta] = s_sum[0];
    }
"""


_iso_phase1_kernel = None
_iso_phase2_sparse_kernel = None


def iso_fused_sparse_attend(
    query: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    q_L: mx.array,
    q_R: Optional[mx.array],
    scale: float,
    dim: int,
    bits: int,
    topk: int = 1024,
) -> mx.array:
    """Two-pass sparse attention over iso-quantized K, V.

    Mirrors `fused_sparse_attend` from the planar module: phase1 scores all
    tokens with QK + iso decompress on K, the bridge picks a per-head
    threshold equal to the topk-th highest score, phase2 does online
    softmax + V iso decompress + accumulate only for tokens at or above
    the threshold. Output shape and LSE merge match `iso_flash_decode`,
    so the two are drop-in interchangeable at the call site.

    When `topk >= seq_len`, the threshold ends up at the minimum score
    so phase2 visits every token and the result is numerically identical
    to `iso_flash_decode` (modulo a difference in the score-write order
    inside fp32 — usually < 1e-4).

    Args:
        query, k_packed, k_norms, v_packed, v_norms, centroids, q_L, q_R,
        scale, dim, bits: same as `iso_flash_decode`.
        topk: keep only the top-k highest-scoring tokens per head.

    Returns: (B, H, 1, dim).
    """
    global _iso_phase1_kernel, _iso_phase2_sparse_kernel
    if _iso_phase1_kernel is None:
        _iso_phase1_kernel = mx.fast.metal_kernel(
            name="iso_phase1_score",
            input_names=["query", "k_packed", "k_norms", "centroids",
                         "q_L", "q_R", "scale", "dims"],
            output_names=["all_scores", "tile_top_scores"],
            source=ISO_PHASE1_SCORE_KERNEL,
        )
    if _iso_phase2_sparse_kernel is None:
        _iso_phase2_sparse_kernel = mx.fast.metal_kernel(
            name="iso_phase2_sparse_attend",
            input_names=["all_scores", "v_packed", "v_norms", "centroids",
                         "q_L", "q_R", "threshold", "dims"],
            output_names=["partial_o", "tile_max", "tile_sum_exp"],
            source=ISO_PHASE2_SPARSE_ATTEND_KERNEL,
        )

    B = query.shape[0]
    H = query.shape[1]
    seq_len = k_norms.shape[2]
    p_dim = k_packed.shape[-1]
    vpw = _VALS_PER_WORD[bits]
    num_tiles = (seq_len + ISO_TILE_SIZE - 1) // ISO_TILE_SIZE
    n_bh = B * H
    top_per_tile = 4
    assert dim % 4 == 0 and dim <= 1024

    has_qR = 1 if q_R is not None else 0
    q_L_flat, q_R_flat, n_heads_real = _prepare_rotors_for_kernel(q_L, q_R, dim)

    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array(
        [dim, seq_len, n_bh, bits, vpw, p_dim, ISO_TILE_SIZE, has_qR, n_heads_real],
        dtype=mx.uint32,
    )

    # ── Phase 1: score all tokens, collect per-tile top-4 ────────────────
    phase1_out = _iso_phase1_kernel(
        inputs=[
            query.astype(mx.float32).reshape(n_bh * dim),
            k_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_bh * seq_len),
            centroids, q_L_flat, q_R_flat, scale_arr, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, n_bh, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_bh * seq_len,),
                       (num_tiles * n_bh * top_per_tile,)],
        output_dtypes=[mx.float32, mx.float32],
    )

    all_scores = phase1_out[0]
    tile_tops = phase1_out[1].reshape(num_tiles, n_bh, top_per_tile)

    # ── Bridge: per-head threshold = topk-th highest score ───────────────
    # tile_tops carries the top-4 from each tile. Per head we have
    # num_tiles * 4 candidates; the topk-th highest among those is the
    # threshold. When topk >= seq_len we must keep every token — but
    # min(all_tops) is *not* "keep everything" (it only thresholds out
    # below-top-4-per-tile tokens), so set threshold to -inf instead.
    all_tops = tile_tops.reshape(-1, n_bh).transpose()  # (n_bh, num_tiles*4)
    n_candidates = all_tops.shape[1]
    if topk >= seq_len:
        threshold = mx.full((n_bh,), -1e30, dtype=mx.float32)
    elif topk < n_candidates:
        topk_vals = mx.topk(all_tops, k=topk, axis=-1)
        threshold = mx.min(topk_vals, axis=-1)
    else:
        # topk between n_candidates and seq_len: not enough tile-top
        # candidates to derive an exact threshold; min(all_tops) is the
        # best conservative estimate (may keep a few more than topk).
        threshold = mx.min(all_tops, axis=-1)
    mx.eval(threshold)  # tiny array — flush before phase2 dispatch

    # ── Phase 2: sparse V decompress + online softmax ────────────────────
    phase2_out = _iso_phase2_sparse_kernel(
        inputs=[
            all_scores,
            v_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(n_bh * seq_len),
            centroids, q_L_flat, q_R_flat, threshold, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, n_bh, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(num_tiles * n_bh * dim,),
                       (num_tiles * n_bh,),
                       (num_tiles * n_bh,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    partial_o = phase2_out[0].reshape(num_tiles, n_bh, dim)
    t_max = phase2_out[1].reshape(num_tiles, n_bh, 1)
    t_sum_exp = phase2_out[2].reshape(num_tiles, n_bh, 1)

    # LSE merge — identical to iso_flash_decode.
    global_max = mx.max(t_max, axis=0, keepdims=True)
    corrections = mx.exp(t_max - global_max)
    numerator = mx.sum(partial_o * corrections, axis=0)
    denominator = mx.sum(t_sum_exp * corrections, axis=0)
    result = numerator / (denominator + 1e-8)

    return result.reshape(B, H, 1, dim)


def iso_unrotate_metal(
    x: mx.array,
    q_L: mx.array,
    q_R: Optional[mx.array] = None,
    rows_per_head: int = 1,
) -> mx.array:
    """Metal-fused inverse IsoQuant rotation. Equivalent to `iso_unrotate` but
    runs a single Metal threadgroup per row.

    Args:
        x: (..., d) — flattens the leading dims into row index for the kernel.
        q_L: (n_groups, 4) for legacy single-rotor mode, or (H, n_groups, 4)
            for per-head rotors. q_R same shape or None.
        rows_per_head: ignored when q_L is 2D. When q_L is 3D the kernel uses
            (row // rows_per_head) % H to pick the rotor for each row, so the
            caller must lay out x in (..., H, T, d) and pass rows_per_head=T.
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
    n_groups = q_L.shape[-2] if q_L.ndim == 3 else q_L.shape[0]
    assert d == n_groups * 4, f"d={d} must equal n_groups*4 ({n_groups * 4})"
    assert d <= 1024, "ISO_INVERSE_ROTATE_KERNEL shared-memory layout assumes d ≤ 1024"

    original_shape = x.shape
    n_rows = 1
    for s in original_shape[:-1]:
        n_rows *= s

    x_flat = x.astype(mx.float32).reshape(n_rows, d)
    q_L_flat, q_R_flat, n_heads_real = _prepare_rotors_for_kernel(q_L, q_R, d)
    has_qR = 1 if q_R is not None else 0
    dims_arr = mx.array(
        [d, n_groups, has_qR, n_heads_real, rows_per_head], dtype=mx.uint32,
    )

    outputs = _iso_inverse_rotate_kernel(
        inputs=[x_flat, q_L_flat, q_R_flat, dims_arr],
        template=[("T", mx.float32)],
        grid=(n_rows * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(n_rows * d,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(original_shape).astype(x.dtype)
