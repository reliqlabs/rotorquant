"""mlx-vlm / mlx-lm compatible KV cache backed by IsoQuant packed storage.

Drop-in replacement for `mlx_lm.models.cache.KVCache`. Stores per-token K and
V vectors as 3-bit (or 2/4-bit) quaternion-rotated packed indices plus a tiny
per-token norm. On read, decompresses lazily via the pure-MLX
`iso_decompress` path (or Metal-fused once those kernels land).

Memory: K/V cache shrinks ~4× vs bf16 at 3-bit. For Leanstral at 128k context
that's roughly 18 GB → 4.5 GB per cache, which is the difference between
running a 4-bit Leanstral on 128 GB and OOM'ing.

Per-layer rotors come from the Modal calibration job. For untuned use,
pass `q_L`/`q_R` generated with `make_random_quaternions` — quality is
worse than calibrated but still functional.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx

from .mlx_fused_iso_attention import (
    _ISO_CODEBOOKS,
    compute_codebooks,
    iso_compress,
    iso_decompress,
)


class IsoKVCache:
    """Iso-quantized K + V cache for one attention layer.

    API mirrors `mlx_lm.models.cache.KVCache` so the cache can be passed to
    `model(..., cache=[IsoKVCache(...), ...])` unchanged. The model sees
    decompressed K, V at full precision each step — compression is invisible
    to the attention math (apart from quantization noise).

    Args:
        bits: 1..4 bits per quantized index.
        q_L, q_R: quaternion rotors (n_groups, 4). `q_R=None` -> SO(3) fast mode.
        head_dim: must equal q_L.shape[0] * 4.
        centroids: optional override for the Lloyd-Max codebook (defaults to
            the precomputed d=128 grid; pass a per-d grid for other head_dims).
    """

    # Class-level sentinel so `hasattr(cache, "bits")` is False — mlx-lm's
    # SDPA uses that as a "this is a quantized cache" signal and would route
    # us to its quantize_matmul path, which expects packed format. Our cache
    # already decompresses to full precision in update_and_fetch.
    # Keep the user-facing API as `iso_bits` to avoid the collision.
    iso_bits: int

    def __init__(
        self,
        bits: int,
        q_L: mx.array,
        q_R: Optional[mx.array],
        head_dim: int,
        centroids: Optional[mx.array] = None,
    ):
        self.iso_bits = bits
        self.q_L = q_L
        self.q_R = q_R
        self.head_dim = head_dim
        assert q_L.shape[0] * 4 == head_dim, (
            f"q_L groups {q_L.shape[0]} × 4 ≠ head_dim {head_dim}"
        )
        if centroids is not None:
            self.centroids = centroids
        elif head_dim in _ISO_CODEBOOKS:
            self.centroids = _ISO_CODEBOOKS[bits]
        else:
            self.centroids = compute_codebooks(head_dim, bits_list=(bits,))[bits]

        # Per-(layer, head, token) packed storage. Built lazily on first update.
        # Shapes after first update: k_packed (B, H, T, packed_dim) uint32
        #                           k_norms  (B, H, T)              float32
        self.k_packed: Optional[mx.array] = None
        self.k_norms: Optional[mx.array] = None
        self.v_packed: Optional[mx.array] = None
        self.v_norms: Optional[mx.array] = None
        self.offset = 0  # number of tokens cached so far

    @property
    def state(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Return the decompressed (K, V) tensors. Used by mlx-lm helpers
        that introspect cache state for cross-step bookkeeping."""
        if self.offset == 0:
            return None, None
        return self._decompress_k(), self._decompress_v()

    @property
    def keys(self) -> Optional[mx.array]:
        return self._decompress_k() if self.offset > 0 else None

    @property
    def values(self) -> Optional[mx.array]:
        return self._decompress_v() if self.offset > 0 else None

    def _compress_block(self, K: mx.array) -> Tuple[mx.array, mx.array]:
        """K shape: (B, H, T, D) -> (packed (B, H, T, packed_dim), norms (B, H, T))."""
        B, H, T, D = K.shape
        flat = K.reshape(-1, D)
        packed, norms = iso_compress(flat, self.iso_bits, self.q_L, self.q_R, self.centroids)
        packed_dim = packed.shape[-1]
        return packed.reshape(B, H, T, packed_dim), norms.reshape(B, H, T)

    def _decompress_k(self) -> mx.array:
        return self._decompress(self.k_packed, self.k_norms)

    def _decompress_v(self) -> mx.array:
        return self._decompress(self.v_packed, self.v_norms)

    def _decompress(self, packed: mx.array, norms: mx.array) -> mx.array:
        """(B, H, T, packed_dim) + (B, H, T) -> (B, H, T, head_dim)."""
        B, H, T, _ = packed.shape
        flat_packed = packed.reshape(-1, packed.shape[-1])
        flat_norms = norms.reshape(-1)
        flat_out = iso_decompress(
            flat_packed, flat_norms, self.head_dim, self.iso_bits,
            self.q_L, self.q_R, self.centroids,
        )
        return flat_out.reshape(B, H, T, self.head_dim)

    def update_and_fetch(self, K: mx.array, V: mx.array) -> Tuple[mx.array, mx.array]:
        """Append new K, V tokens; return the full decompressed (K, V) so far.

        K, V shape: (B, n_kv_heads, T_new, head_dim).
        Returns the same shape but with T = total tokens cached.
        """
        new_k_packed, new_k_norms = self._compress_block(K)
        new_v_packed, new_v_norms = self._compress_block(V)

        if self.k_packed is None:
            self.k_packed = new_k_packed
            self.k_norms = new_k_norms
            self.v_packed = new_v_packed
            self.v_norms = new_v_norms
        else:
            self.k_packed = mx.concatenate([self.k_packed, new_k_packed], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, new_k_norms], axis=2)
            self.v_packed = mx.concatenate([self.v_packed, new_v_packed], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, new_v_norms], axis=2)

        self.offset += K.shape[2]
        return self._decompress_k(), self._decompress_v()

    def memory_bytes(self) -> int:
        """Storage footprint in bytes (packed indices + norms)."""
        if self.k_packed is None:
            return 0
        return (
            self.k_packed.nbytes + self.k_norms.nbytes
            + self.v_packed.nbytes + self.v_norms.nbytes
        )


def load_rotors_into_cache_factory(rotors_path: str, head_dim: int, bits: int = 3):
    """Read a `rotors.safetensors` produced by Modal calibration and return a
    factory `(layer_idx: int) -> IsoKVCache | None`.

    Use:
        cache_factory = load_rotors_into_cache_factory("rotors.safetensors", 128, 3)
        caches = [cache_factory(i) or KVCache() for i in range(model.n_layers)]

    Layers not present in the rotors file fall back to `None` so the caller
    can substitute a standard KVCache for those layers.
    """
    try:
        loaded = mx.load(rotors_path)
    except Exception:
        from safetensors.numpy import load_file
        loaded = {k: mx.array(v) for k, v in load_file(rotors_path).items()}

    rotors_by_layer: dict[int, dict[str, mx.array]] = {}
    for key, tensor in loaded.items():
        # Expected keys: "layer_<N>.q_L" and (optionally) "layer_<N>.q_R"
        try:
            layer_str, which = key.split(".")
            li = int(layer_str.removeprefix("layer_"))
        except (ValueError, AttributeError):
            continue
        rotors_by_layer.setdefault(li, {})[which] = tensor

    def factory(layer_idx: int) -> Optional[IsoKVCache]:
        if layer_idx not in rotors_by_layer:
            return None
        entry = rotors_by_layer[layer_idx]
        return IsoKVCache(
            bits=bits,
            q_L=entry["q_L"],
            q_R=entry.get("q_R"),
            head_dim=head_dim,
        )

    return factory
