"""Gradient-based rotor refinement for IsoQuant.

Pure-torch, lives outside the Modal-only `modal_apps/` tree so we can
import + unit-test it from this repo's normal test suite. The Modal
calibration app re-exports it through a thin wrapper.

The trick is the straight-through estimator (STE) on the argmin
quantize: forward is the hard nearest-centroid lookup, backward
treats it as identity so gradients flow into the quaternion parameters.

Multi-start support: try N different random starting quaternions in
parallel, run Adam on each, return the best final result. This is the
fix for the "Adam stuck at random-search local min" problem — random
search at 64 seeds finds a decent local minimum; without restarting
Adam from somewhere else, you just refine the same minimum and gain
little. With N=8 restarts we typically lift cos sim by 0.01-0.02 on
realistic data.
"""

from __future__ import annotations

from typing import Optional

import torch


# ── Quaternion primitives (torch) ───────────────────────────────────────────


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (w, -x, -y, -z)."""
    signs = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    return q * signs


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion tensors (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([rw, rx, ry, rz], dim=-1)


def make_random_unit_quats(n_groups: int, generator: torch.Generator,
                           device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Generate (n_groups, 4) random unit quaternions on a normal-then-normalize path."""
    q = torch.randn(n_groups, 4, generator=generator, dtype=dtype)
    q = q.to(device)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ── Forward + STE quantize ──────────────────────────────────────────────────


def _iso_forward_ste(K_blocks: torch.Tensor, q_L: torch.Tensor,
                     q_R: Optional[torch.Tensor], centroids: torch.Tensor) -> torch.Tensor:
    """Differentiable quantize/dequantize: rotate -> hard-nearest -> STE -> unrotate.

    K_blocks: (N, n_groups, 4) — unit-norm K vectors split into 4-blocks.
    q_L: (n_groups, 4), q_R: (n_groups, 4) or None.
    centroids: (n_levels,) Lloyd-Max grid.
    Returns: (N, n_groups*4) — reconstructed (in unit space).
    """
    qL = q_L.unsqueeze(0)  # broadcast over batch
    tmp = quat_mul(qL, K_blocks)
    if q_R is not None:
        rot = quat_mul(tmp, quat_conj(q_R).unsqueeze(0))
    else:
        rot = tmp

    flat = rot.reshape(rot.shape[0], -1)  # (N, n_groups*4)
    diffs = (flat.unsqueeze(-1) - centroids).abs()  # (N, d, n_levels)
    idx = diffs.argmin(dim=-1)
    hard = centroids[idx]
    # STE: forward hard, backward identity.
    q_flat = flat + (hard - flat).detach()
    q_blocks = q_flat.view(rot.shape)

    if q_R is not None:
        tmp2 = quat_mul(quat_conj(q_L).unsqueeze(0), q_blocks)
        recon = quat_mul(tmp2, q_R.unsqueeze(0))
    else:
        recon = quat_mul(quat_conj(q_L).unsqueeze(0), q_blocks)
    return recon.reshape(recon.shape[0], -1)


def _iso_forward_soft(K_blocks: torch.Tensor, q_L: torch.Tensor,
                      q_R: Optional[torch.Tensor], centroids: torch.Tensor,
                      temperature: float = 0.05) -> torch.Tensor:
    """Soft variant of the quantize step — replaces argmin with a temperature-
    weighted softmax over centroids. As temperature → 0, behavior approaches
    hard quantize; at the temperature we use (0.05), it's smooth enough to
    propagate meaningful gradients into q_L / q_R while staying close to the
    hard reconstruction.

    Better for optimization than STE because the gradient direction actually
    corresponds to reducing the loss surface (STE gives a 'lying' gradient
    that points where the loss would go if quantize were identity — which it
    isn't, so the optimizer hits flat local minima).
    """
    qL = q_L.unsqueeze(0)
    tmp = quat_mul(qL, K_blocks)
    if q_R is not None:
        rot = quat_mul(tmp, quat_conj(q_R).unsqueeze(0))
    else:
        rot = tmp

    flat = rot.reshape(rot.shape[0], -1)
    # Weighted sum over centroids: soft analog of `centroids[argmin]`.
    neg_dist = -(flat.unsqueeze(-1) - centroids).pow(2) / temperature
    weights = torch.softmax(neg_dist, dim=-1)  # (N, d, n_levels)
    soft = (weights * centroids).sum(dim=-1)  # (N, d)
    q_blocks = soft.view(rot.shape)

    if q_R is not None:
        tmp2 = quat_mul(quat_conj(q_L).unsqueeze(0), q_blocks)
        recon = quat_mul(tmp2, q_R.unsqueeze(0))
    else:
        recon = quat_mul(quat_conj(q_L).unsqueeze(0), q_blocks)
    return recon.reshape(recon.shape[0], -1)


# ── Single-start gradient refinement ────────────────────────────────────────


def _adam_refine_single(K_blocks: torch.Tensor, head_dim: int, mode: str,
                        centroids: torch.Tensor, init_q_L: torch.Tensor,
                        init_q_R: Optional[torch.Tensor],
                        n_steps: int, lr: float, loss_kind: str,
                        forward_kind: str = "soft",
                        soft_temperature: float = 0.05) -> dict:
    """Adam loop from a fixed init. Returns dict with refined q_L, q_R and final cos.

    `forward_kind`:
      - 'soft': softmax-weighted centroids during training (default; works).
      - 'ste':  hard quantize with straight-through estimator (kept for
        comparison; tends to stall at random-search local min).

    Evaluation always uses the hard forward — soft is purely for gradients.
    """
    q_L = init_q_L.clone().detach().requires_grad_(True)
    params = [q_L]
    q_R = None
    if mode == "full":
        q_R = init_q_R.clone().detach().requires_grad_(True)
        params.append(q_R)

    opt = torch.optim.Adam(params, lr=lr)

    def fwd(qL, qR):
        if forward_kind == "ste":
            return _iso_forward_ste(K_blocks, qL, qR, centroids)
        return _iso_forward_soft(K_blocks, qL, qR, centroids,
                                 temperature=soft_temperature)

    K_flat = K_blocks.reshape(K_blocks.shape[0], -1)

    for _ in range(n_steps):
        opt.zero_grad()
        K_hat = fwd(q_L, q_R)
        if loss_kind == "cos":
            nx = K_flat.norm(dim=-1).clamp_min(1e-8)
            ny = K_hat.norm(dim=-1).clamp_min(1e-8)
            cos = (K_flat * K_hat).sum(dim=-1) / (nx * ny)
            loss = 1.0 - cos.mean()
        else:
            loss = (K_flat - K_hat).pow(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q_L.div_(q_L.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            if q_R is not None:
                q_R.div_(q_R.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    # Final evaluation: ALWAYS hard quantize, regardless of training forward.
    # The hard cos is what production inference cares about.
    with torch.no_grad():
        K_hat_hard = _iso_forward_ste(K_blocks, q_L, q_R, centroids)
        nx = K_flat.norm(dim=-1).clamp_min(1e-8)
        ny = K_hat_hard.norm(dim=-1).clamp_min(1e-8)
        cos = (K_flat * K_hat_hard).sum(dim=-1) / (nx * ny)
        return {
            "cos_mean": cos.mean().item(),
            "cos_p05": cos.quantile(0.05).item(),
            "mse": (K_flat - K_hat_hard).pow(2).mean().item(),
            "q_L": q_L.detach().cpu().clone(),
            "q_R": (q_R.detach().cpu().clone() if q_R is not None else None),
        }


# ── Multi-start refinement (the actual public entry) ────────────────────────


def gradient_refine_rotors(K: torch.Tensor, head_dim: int, bits: int, mode: str,
                           init: dict, n_steps: int = 300, lr: float = 1e-2,
                           n_inits: int = 8, loss_kind: str = "cos",
                           forward_kind: str = "soft",
                           soft_temperature: float = 0.05,
                           centroid_lloyd_max_fn=None, seed: int = 12345) -> dict:
    """Multi-start Adam refinement of IsoQuant rotors.

    Args:
        K: (N, head_dim) tensor of K vectors captured from the model.
        head_dim: head_dim (= n_groups * 4).
        bits: quantization bits.
        mode: 'full' or 'fast'.
        init: dict with keys 'q_L' and (if mode='full') 'q_R' — the best
              result of a random search; serves as restart #0.
        n_steps: Adam iterations per init.
        lr: Adam learning rate.
        n_inits: total number of initializations to try, including `init`.
                 Extra inits are sampled as random unit quaternions.
        loss_kind: 'cos' (1 - cos sim) or 'mse'.
        centroid_lloyd_max_fn: callable d, bits -> centroid tensor. Default
            uses turboquant.lloyd_max.solve_lloyd_max.
    Returns: same dict shape as `_random_rotor_search`, refined.
    """
    if centroid_lloyd_max_fn is None:
        from turboquant.lloyd_max import solve_lloyd_max  # local import
        centroids_torch, _ = solve_lloyd_max(head_dim, bits)
    else:
        centroids_torch = centroid_lloyd_max_fn(head_dim, bits)

    device = K.device
    centroids = centroids_torch.to(device)

    K = K.to(torch.float32)
    norms = K.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K_unit = K / norms
    n_groups = head_dim // 4
    K_blocks = K_unit.view(K_unit.shape[0], n_groups, 4)

    # init #0 from the random search
    inits = [{
        "q_L": init["q_L"].to(device),
        "q_R": init["q_R"].to(device) if (mode == "full" and init.get("q_R") is not None) else None,
    }]
    # extra random inits
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for _ in range(n_inits - 1):
        qL = make_random_unit_quats(n_groups, gen, device)
        qR = make_random_unit_quats(n_groups, gen, device) if mode == "full" else None
        inits.append({"q_L": qL, "q_R": qR})

    best = None
    for init_idx, init_pair in enumerate(inits):
        result = _adam_refine_single(
            K_blocks, head_dim, mode, centroids,
            init_pair["q_L"], init_pair["q_R"],
            n_steps=n_steps, lr=lr, loss_kind=loss_kind,
            forward_kind=forward_kind, soft_temperature=soft_temperature,
        )
        if best is None or result["cos_mean"] > best["cos_mean"]:
            best = result
            best["start_idx"] = init_idx

    return best
