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
    _DEFAULT_D,
    _ISO_CODEBOOKS,
    compute_codebooks,
    iso_compress,
    iso_decompress,
    iso_flash_decode,
    iso_fused_sparse_attend,
)


# Cache codebooks by (head_dim, bits) so each constructor call is O(1) after
# the first. Lloyd-Max via scipy.integrate.quad is ~5-30s per (d, bits) combo
# and rebuilding it on every IsoKVCache instance was a hidden perf cliff.
_CODEBOOK_CACHE: dict[tuple[int, int], mx.array] = {}


def _get_codebook(head_dim: int, bits: int) -> mx.array:
    key = (head_dim, bits)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    if head_dim == _DEFAULT_D and bits in _ISO_CODEBOOKS:
        cb = _ISO_CODEBOOKS[bits]
    else:
        cb = compute_codebooks(head_dim, bits_list=(bits,))[bits]
    _CODEBOOK_CACHE[key] = cb
    return cb


class IsoKVCache:
    """Iso-quantized K + V cache for one attention layer.

    API mirrors `mlx_lm.models.cache.KVCache` so the cache can be passed to
    `model(..., cache=[IsoKVCache(...), ...])` unchanged. The model sees
    decompressed K, V at full precision each step — compression is invisible
    to the attention math (apart from quantization noise).

    Args:
        bits: 1..4 bits per quantized index.
        q_L, q_R: quaternion rotors. Either `(n_groups, 4)` for a single
            rotor broadcast across every head (legacy), OR
            `(n_heads, n_groups, 4)` for per-head rotors. `q_R=None` ->
            SO(3) fast mode regardless of q_L shape.
        head_dim: must equal q_L.shape[-2] * 4 (3D q_L) or q_L.shape[0] * 4 (2D).
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
        self.per_head_rotors = q_L.ndim == 3
        n_groups = q_L.shape[-2] if self.per_head_rotors else q_L.shape[0]
        assert n_groups * 4 == head_dim, (
            f"q_L shape {tuple(q_L.shape)} × 4 ≠ head_dim {head_dim}"
        )
        if self.per_head_rotors and q_R is not None:
            assert q_R.shape == q_L.shape, (
                f"per-head q_L shape {tuple(q_L.shape)} != "
                f"q_R shape {tuple(q_R.shape)}"
            )
        # `head_dim in _ISO_CODEBOOKS` was always False before (the dict is
        # keyed by bits) — falling through to compute_codebooks on every call.
        # Now goes via the (head_dim, bits) cache so repeated construction
        # (e.g., one cache per layer) is free.
        if centroids is not None:
            self.centroids = centroids
        else:
            self.centroids = _get_codebook(head_dim, bits)

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
        """K shape: (B, H, T, D) -> (packed (B, H, T, packed_dim), norms (B, H, T)).

        For per-head rotors we transpose so heads end up at axis -2 inside
        iso_compress (which the underlying iso_rotate requires for broadcast):
            (B, H, T, D) -> transpose -> (B, T, H, D) -> reshape -> (B*T, H, D).
        After compression we undo the transpose so storage stays (B, H, T, ...).
        """
        B, H, T, D = K.shape
        if self.per_head_rotors:
            K_perm = K.transpose(0, 2, 1, 3).reshape(B * T, H, D)
            packed, norms = iso_compress(K_perm, self.iso_bits, self.q_L,
                                          self.q_R, self.centroids)
            packed_dim = packed.shape[-1]
            packed = packed.reshape(B, T, H, packed_dim).transpose(0, 2, 1, 3)
            norms = norms.reshape(B, T, H).transpose(0, 2, 1)
            return packed, norms
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
        if self.per_head_rotors:
            packed_perm = packed.transpose(0, 2, 1, 3).reshape(B * T, H, packed.shape[-1])
            norms_perm = norms.transpose(0, 2, 1).reshape(B * T, H)
            out = iso_decompress(
                packed_perm, norms_perm, self.head_dim, self.iso_bits,
                self.q_L, self.q_R, self.centroids,
            )
            return out.reshape(B, T, H, self.head_dim).transpose(0, 2, 1, 3)
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

    def attend(
        self,
        query: mx.array,
        scale: float,
        topk: Optional[int] = None,
    ) -> mx.array:
        """Compute scaled-dot-product attention directly against the packed
        cache via the fused Metal kernel — no decompression detour.

        Bypasses the `decompress -> mx.fast.scaled_dot_product_attention`
        path that `update_and_fetch` returns, taking the ~8-13x speedup
        documented in the README at T >= 1024. Use this in custom attention
        layers when you control the SDPA call site; the standard mlx-lm
        attention will route through update_and_fetch + SDPA instead, which
        still works but pays the decompression cost.

        Args:
            query: (B, H, 1, head_dim) — current decode-step query.
            scale: softmax temperature (1/sqrt(head_dim) typically).
            topk: if set, route to the two-pass sparse path keeping only the
                topk highest-scoring tokens per head. When `topk >= offset`
                (i.e., not enough tokens to drop any), this still routes
                through sparse_attend but with threshold=-inf so the result
                matches dense flash decode within fp32 rounding. Leave None
                to use the dense one-pass flash decode.

        Returns:
            (B, H, 1, head_dim) attention output.
        """
        if self.k_packed is None:
            raise RuntimeError("IsoKVCache.attend called before update_and_fetch")
        if topk is not None:
            return iso_fused_sparse_attend(
                query=query,
                k_packed=self.k_packed,
                k_norms=self.k_norms,
                v_packed=self.v_packed,
                v_norms=self.v_norms,
                centroids=self.centroids,
                q_L=self.q_L,
                q_R=self.q_R,
                scale=scale,
                dim=self.head_dim,
                bits=self.iso_bits,
                topk=topk,
            )
        return iso_flash_decode(
            query=query,
            k_packed=self.k_packed,
            k_norms=self.k_norms,
            v_packed=self.v_packed,
            v_norms=self.v_norms,
            centroids=self.centroids,
            q_L=self.q_L,
            q_R=self.q_R,
            scale=scale,
            dim=self.head_dim,
            bits=self.iso_bits,
        )


def load_rotors_into_cache_factory(rotors_path: str, head_dim: int, bits: int = 3):
    """Read a `rotors.safetensors` produced by Modal calibration and return a
    factory `(layer_idx: int) -> IsoKVCache | None`.

    Supports two layouts:
      - per-layer (legacy): keys `layer_<N>.q_L` / `layer_<N>.q_R`. One rotor
        per layer, broadcast across all heads. q_L shape (n_groups, 4).
      - per-head: keys `layer_<N>_head_<H>.q_L` / `layer_<N>_head_<H>.q_R`.
        Stacked into `(n_heads, n_groups, 4)` before being handed to the cache.
        Heads missing from the file get a zero-rotor row that the kernel
        will treat as identity-ish; this is a calibration bug if it ever
        happens so we warn instead.

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

    per_layer: dict[int, dict[str, mx.array]] = {}
    per_head: dict[int, dict[int, dict[str, mx.array]]] = {}
    for key, tensor in loaded.items():
        # Try the per-head pattern first: layer_<N>_head_<H>.q_{L,R}
        try:
            stem, which = key.split(".")
        except ValueError:
            continue
        if "_head_" in stem:
            try:
                layer_part, head_part = stem.split("_head_")
                li = int(layer_part.removeprefix("layer_"))
                hi = int(head_part)
            except ValueError:
                continue
            per_head.setdefault(li, {}).setdefault(hi, {})[which] = tensor
        elif stem.startswith("layer_"):
            try:
                li = int(stem.removeprefix("layer_"))
            except ValueError:
                continue
            per_layer.setdefault(li, {})[which] = tensor

    def _stack_per_head(layer_idx: int) -> tuple[mx.array, Optional[mx.array]]:
        entries = per_head[layer_idx]
        n_heads = max(entries) + 1
        q_L_list, q_R_list, has_qR = [], [], False
        for hi in range(n_heads):
            e = entries.get(hi)
            if e is None or "q_L" not in e:
                raise ValueError(
                    f"rotors file missing layer_{layer_idx}_head_{hi}.q_L"
                )
            q_L_list.append(e["q_L"])
            qR = e.get("q_R")
            if qR is not None:
                has_qR = True
                q_R_list.append(qR)
            else:
                q_R_list.append(None)
        q_L = mx.stack(q_L_list, axis=0)
        if has_qR:
            if any(x is None for x in q_R_list):
                raise ValueError(
                    f"layer_{layer_idx}: some heads have q_R and others don't"
                )
            q_R = mx.stack(q_R_list, axis=0)
        else:
            q_R = None
        return q_L, q_R

    def factory(layer_idx: int) -> Optional[IsoKVCache]:
        if layer_idx in per_head:
            q_L, q_R = _stack_per_head(layer_idx)
        elif layer_idx in per_layer:
            entry = per_layer[layer_idx]
            q_L = entry["q_L"]
            q_R = entry.get("q_R")
        else:
            return None
        return IsoKVCache(bits=bits, q_L=q_L, q_R=q_R, head_dim=head_dim)

    return factory
