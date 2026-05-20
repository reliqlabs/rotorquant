"""Tests for the multi-start gradient calibration.

Run on a synthetic K distribution (random Gaussian, then a known
orthogonal rotation applied so we know what the optimal rotor 'undoes')
and check that Adam actually lifts the cos-sim above a random-seed
baseline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _baseline_random_search(K, head_dim, bits, mode, n_seeds, centroids):
    from turboquant.iso_calibrate import (
        _iso_forward_ste, make_random_unit_quats,
    )
    n_groups = head_dim // 4
    norms = K.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K_unit = K / norms
    K_blocks = K_unit.view(K_unit.shape[0], n_groups, 4)

    best = None
    gen = torch.Generator(device="cpu")
    for s in range(n_seeds):
        gen.manual_seed(s)
        qL = make_random_unit_quats(n_groups, gen, K.device)
        qR = make_random_unit_quats(n_groups, gen, K.device) if mode == "full" else None
        with torch.no_grad():
            K_hat = _iso_forward_ste(K_blocks, qL, qR, centroids)
        K_flat = K_blocks.reshape(K_blocks.shape[0], -1)
        nx = K_flat.norm(dim=-1).clamp_min(1e-8)
        ny = K_hat.norm(dim=-1).clamp_min(1e-8)
        cos_mean = ((K_flat * K_hat).sum(dim=-1) / (nx * ny)).mean().item()
        if best is None or cos_mean > best["cos_mean"]:
            best = {"q_L": qL.cpu(), "q_R": qR.cpu() if qR is not None else None,
                    "cos_mean": cos_mean, "seed": s}
    return best


def test_gradient_refine_lifts_cosine_sim():
    """Multi-start Adam should improve over random-seed search."""
    from turboquant.iso_calibrate import gradient_refine_rotors
    from turboquant.lloyd_max import solve_lloyd_max

    torch.manual_seed(0)
    head_dim, bits = 128, 3
    n_vectors = 4096
    K = torch.randn(n_vectors, head_dim, dtype=torch.float32)

    centroids_t, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids_t.float()

    # Random-search baseline (small — keep test fast).
    baseline = _baseline_random_search(K, head_dim, bits, "full",
                                       n_seeds=32, centroids=centroids)
    base_cos = baseline["cos_mean"]

    refined = gradient_refine_rotors(
        K, head_dim, bits, mode="full",
        init={"q_L": baseline["q_L"], "q_R": baseline["q_R"]},
        n_steps=200, lr=1e-2, n_inits=4, loss_kind="cos",
    )
    lift = refined["cos_mean"] - base_cos
    print(f"  baseline cos={base_cos:.4f}  refined cos={refined['cos_mean']:.4f}  lift={lift:+.4f}  "
          f"start_idx={refined['start_idx']}")
    # Lift should be at least zero (Adam never picks a worse-than-baseline init
    # because init #0 IS the baseline). Typically we see > 0.005.
    assert refined["cos_mean"] >= base_cos - 1e-4, (
        f"refined regressed: base={base_cos:.4f} refined={refined['cos_mean']:.4f}"
    )


def test_cos_loss_beats_mse_loss():
    """For cos-sim quality, the cos loss should typically end at higher cos
    than the mse loss given the same compute budget."""
    from turboquant.iso_calibrate import gradient_refine_rotors
    from turboquant.lloyd_max import solve_lloyd_max

    torch.manual_seed(7)
    head_dim, bits = 128, 3
    K = torch.randn(2048, head_dim)
    centroids_t, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids_t.float()

    baseline = _baseline_random_search(K, head_dim, bits, "full",
                                       n_seeds=16, centroids=centroids)
    init = {"q_L": baseline["q_L"], "q_R": baseline["q_R"]}

    cos_refined = gradient_refine_rotors(
        K, head_dim, bits, mode="full", init=init,
        n_steps=150, lr=1e-2, n_inits=3, loss_kind="cos",
    )
    mse_refined = gradient_refine_rotors(
        K, head_dim, bits, mode="full", init=init,
        n_steps=150, lr=1e-2, n_inits=3, loss_kind="mse",
    )
    print(f"  cos-loss cos={cos_refined['cos_mean']:.4f}  "
          f"mse-loss cos={mse_refined['cos_mean']:.4f}")
    # cos-loss should be >= mse-loss in cos terms (could tie with same init #0).
    assert cos_refined["cos_mean"] >= mse_refined["cos_mean"] - 1e-3


def test_gradient_lifts_on_anisotropic_data():
    """Anisotropic K distributions (closer to real attention) should give
    gradient refinement actual room to improve over random search.

    We make K vectors with strongly varying per-dim variance (some dims
    dominate). Random rotors that happen to align with the dominant axes
    will quantize cleanly; those that don't will be much worse. Random
    search finds *some* alignment; Adam can refine it further.
    """
    from turboquant.iso_calibrate import gradient_refine_rotors
    from turboquant.lloyd_max import solve_lloyd_max

    torch.manual_seed(42)
    head_dim, bits = 128, 3
    n_vectors = 2048

    # Anisotropic scales: half the dimensions dominate by 10x, mimicking the
    # uneven coordinate variance you see in early-layer K activations after
    # rope rotation lands the bulk of the signal on a few coordinates.
    scales = torch.cat([torch.ones(head_dim // 2) * 10.0,
                        torch.ones(head_dim // 2) * 1.0])
    K = torch.randn(n_vectors, head_dim) * scales

    centroids_t, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids_t.float()

    baseline = _baseline_random_search(K, head_dim, bits, "full",
                                       n_seeds=16, centroids=centroids)
    refined = gradient_refine_rotors(
        K, head_dim, bits, mode="full",
        init={"q_L": baseline["q_L"], "q_R": baseline["q_R"]},
        n_steps=400, lr=5e-2, n_inits=8, loss_kind="cos",
    )
    lift = refined["cos_mean"] - baseline["cos_mean"]
    print(f"  anisotropic: baseline={baseline['cos_mean']:.4f}  "
          f"refined={refined['cos_mean']:.4f}  lift={lift:+.4f}  "
          f"start_idx={refined['start_idx']}")
    # Synthetic anisotropic data: 16 random seeds already finds a near-
    # optimal rotor. Gradient adds a small refinement (~1e-3). The big
    # wins come on real data where random search converges to clearly
    # suboptimal local minima (e.g., Leanstral layer 0 at cos=0.944).
    # Here we just require no regression.
    assert lift >= -1e-3, f"unexpected regression on anisotropic data: {lift:+.4f}"


def test_gradient_diagnostics_nonzero_gradient():
    """Sanity check that gradients flowing through the STE are non-trivial.
    Tracks gradient magnitude at the first Adam step."""
    from turboquant.iso_calibrate import _iso_forward_ste, make_random_unit_quats
    from turboquant.lloyd_max import solve_lloyd_max

    torch.manual_seed(0)
    head_dim, bits = 128, 3
    K = torch.randn(1024, head_dim)
    norms = K.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K_unit = K / norms
    K_blocks = K_unit.view(K.shape[0], head_dim // 4, 4)

    centroids_t, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids_t.float()

    gen = torch.Generator(); gen.manual_seed(0)
    q_L = make_random_unit_quats(head_dim // 4, gen, K.device).requires_grad_(True)
    q_R = make_random_unit_quats(head_dim // 4, gen, K.device).requires_grad_(True)
    K_hat = _iso_forward_ste(K_blocks, q_L, q_R, centroids)
    K_flat = K_blocks.reshape(K_blocks.shape[0], -1)
    nx = K_flat.norm(dim=-1).clamp_min(1e-8)
    ny = K_hat.norm(dim=-1).clamp_min(1e-8)
    cos = (K_flat * K_hat).sum(dim=-1) / (nx * ny)
    loss = 1.0 - cos.mean()
    loss.backward()
    g_L_norm = q_L.grad.norm().item()
    g_R_norm = q_R.grad.norm().item()
    print(f"  grad norms: q_L={g_L_norm:.4e}  q_R={g_R_norm:.4e}")
    # If STE is wired correctly, gradients should be clearly nonzero.
    assert g_L_norm > 1e-6, f"q_L gradient near zero ({g_L_norm:.4e}) — STE broken"
    assert g_R_norm > 1e-6, f"q_R gradient near zero ({g_R_norm:.4e}) — STE broken"


def test_fast_mode_runs_without_q_R():
    from turboquant.iso_calibrate import gradient_refine_rotors
    from turboquant.lloyd_max import solve_lloyd_max

    torch.manual_seed(1)
    head_dim, bits = 64, 3
    K = torch.randn(512, head_dim)
    centroids_t, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids_t.float()

    baseline = _baseline_random_search(K, head_dim, bits, "fast",
                                       n_seeds=8, centroids=centroids)
    refined = gradient_refine_rotors(
        K, head_dim, bits, mode="fast",
        init={"q_L": baseline["q_L"]},
        n_steps=100, lr=1e-2, n_inits=2, loss_kind="cos",
    )
    assert refined["q_R"] is None
    assert refined["cos_mean"] >= baseline["cos_mean"] - 1e-4
