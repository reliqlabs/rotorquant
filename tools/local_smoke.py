"""Local end-to-end smoke for the iso calibration -> inference pipeline.

The Modal calibration run is what produces production rotors for Leanstral,
but it costs real money and runs for hours. Before spinning Modal up again
we want to confirm that the full loop — capture K from a real model,
calibrate rotors, load them into IsoKVCache, run generation — works end
to end. This script does that loop locally on Qwen2.5-0.5B-Instruct-4bit
in ~2 minutes on the M2 24GB.

Pipeline:
  1. Load MLX Qwen-0.5B and run a calibration prompt through it with a
     standard prompt cache.
  2. Pull K out of cache[layer].keys for the layers we want to calibrate,
     convert to torch.
  3. Run multi-start gradient refinement (cos-loss) to produce rotors.
  4. Save rotors.safetensors.
  5. Reload Qwen, run generation 3 ways:
       baseline  — make_prompt_cache (default KVCache)
       iso-random — IsoKVCache with random quaternions
       iso-calibrated — IsoKVCache with our rotors
     Print all three outputs side by side and compute a coarse "did the
     calibrated path stay closer to baseline than random" indicator.

Usage:
  python tools/local_smoke.py
  python tools/local_smoke.py --layers 0,4,8,12,16,20,23 --grad-steps 200

This is *not* a quality eval — Qwen-0.5B can't do Lean, and 64-dim heads
are easier to quantize than Leanstral's 128. The intent is to catch
plumbing bugs (shape mismatches, dtype confusion, save/load layout, MLX
cache wiring) before we burn another Modal hour.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, ".")

DEFAULT_CALIB_TEXT = (
    "The quick brown fox jumps over the lazy dog. Lloyd-Max quantization "
    "optimally allocates centroids by minimizing expected squared error "
    "over the source distribution, given a probability density. When "
    "combined with orthogonal rotation, this lets us quantize key/value "
    "vectors while preserving inner products that drive attention. "
    "Quaternion rotations in 4D form an SO(4) subgroup whose action is "
    "norm-preserving, which is exactly the invariance we need for the "
    "softmax in scaled dot-product attention to remain accurate. Random "
    "rotors give a useful baseline; gradient refinement on captured K "
    "activations lifts cosine similarity another 0.01-0.03 in practice."
) * 4  # ~250 tokens — enough K samples for stable Lloyd-Max + STE.

GENERATION_PROMPT = "The capital of France is"


def _capture_k_per_layer(model, tokenizer, calib_text: str, layer_ids: list[int]) -> dict[int, mx.array]:
    """Run one forward with a default prompt cache, return K per requested layer.
    K shape per layer: (B*H*T, head_dim) as mx.float32, ready for torch."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    ids = mx.array(tokenizer.encode(calib_text))[None, :]
    print(f"[calib] prefilling {ids.shape[1]} tokens", flush=True)
    t0 = time.time()
    _ = model(ids, cache=cache)
    mx.eval(_)
    print(f"[calib] forward pass in {time.time() - t0:.1f}s", flush=True)

    out: dict[int, mx.array] = {}
    for li in layer_ids:
        if li >= len(cache):
            print(f"[calib]   layer {li}: out of range (n_layers={len(cache)}), skip",
                  flush=True)
            continue
        K = cache[li].keys  # (B, H, T_padded, head_dim)
        t_actual = cache[li].offset
        K = K[:, :, :t_actual, :].astype(mx.float32)
        n = K.shape[0] * K.shape[1] * K.shape[2]
        out[li] = K.reshape(n, K.shape[-1])
        print(f"[calib]   layer {li}: K (B,H,T,D)={tuple(K.shape)} -> {n} vectors",
              flush=True)
    return out


def _calibrate_layer(K_mx: mx.array, head_dim: int, bits: int, mode: str,
                      n_seeds: int, grad_steps: int, n_inits: int) -> dict:
    """Random search seed + multi-start gradient refine in torch.

    Mirrors the modal calibrator but in-process (no Modal). K comes in as
    an MLX array; we convert to torch for the calibration math.
    """
    import numpy as np
    import torch
    from turboquant.iso_calibrate import gradient_refine_rotors, make_random_unit_quats
    from turboquant.iso_calibrate import _iso_forward_ste
    from turboquant.lloyd_max import solve_lloyd_max

    K = torch.from_numpy(np.asarray(K_mx, copy=False)).float()
    centroids_t, _ = solve_lloyd_max(head_dim, bits)
    centroids = centroids_t.float()

    # 1) Random-rotor seed search to pick a good init for Adam.
    n_groups = head_dim // 4
    norms = K.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K_unit = K / norms
    K_blocks = K_unit.view(K.shape[0], n_groups, 4)
    K_flat = K_blocks.reshape(K.shape[0], -1)

    best = None
    gen = torch.Generator(device="cpu")
    for s in range(n_seeds):
        gen.manual_seed(s)
        qL = make_random_unit_quats(n_groups, gen, K.device)
        qR = make_random_unit_quats(n_groups, gen, K.device) if mode == "full" else None
        with torch.no_grad():
            K_hat = _iso_forward_ste(K_blocks, qL, qR, centroids)
        nx = K_flat.norm(dim=-1).clamp_min(1e-8)
        ny = K_hat.norm(dim=-1).clamp_min(1e-8)
        cos_mean = ((K_flat * K_hat).sum(dim=-1) / (nx * ny)).mean().item()
        if best is None or cos_mean > best["cos_mean"]:
            best = {"q_L": qL.cpu(), "q_R": qR.cpu() if qR is not None else None,
                    "cos_mean": cos_mean, "seed": s}

    # 2) Multi-start gradient refinement from the random-search best.
    refined = gradient_refine_rotors(
        K, head_dim, bits, mode=mode,
        init={"q_L": best["q_L"], "q_R": best["q_R"]},
        n_steps=grad_steps, lr=1e-2, n_inits=n_inits,
        loss_kind="cos", forward_kind="soft", soft_temperature=0.05,
    )
    return {
        "q_L": refined["q_L"],
        "q_R": refined["q_R"],
        "cos_random": best["cos_mean"],
        "cos_refined": refined["cos_mean"],
    }


def _save_rotors(rotors_path: Path, rotor_state: dict[str, "torch.Tensor"]) -> None:
    from safetensors.torch import save_file
    save_file(rotor_state, str(rotors_path), metadata={"format": "pt"})


def _build_iso_caches(n_layers: int, head_dim: int, bits: int, mode: str,
                      rotors_path: Path | None) -> list:
    from turboquant.iso_kv_cache import IsoKVCache, load_rotors_into_cache_factory
    from turboquant.mlx_fused_iso_attention import make_random_quaternions

    factory = (load_rotors_into_cache_factory(str(rotors_path), head_dim, bits)
               if rotors_path else (lambda li: None))
    fallback_q_L = make_random_quaternions(head_dim // 4, seed=1)
    fallback_q_R = make_random_quaternions(head_dim // 4, seed=2) if mode == "full" else None
    caches = []
    n_cal = 0
    for li in range(n_layers):
        c = factory(li)
        if c is None:
            c = IsoKVCache(bits=bits, q_L=fallback_q_L,
                           q_R=fallback_q_R, head_dim=head_dim)
        else:
            n_cal += 1
        caches.append(c)
    print(f"[gen] built {n_layers} iso caches ({n_cal} calibrated, "
          f"{n_layers - n_cal} fallback)", flush=True)
    return caches


def _generate(model, tokenizer, prompt: str, cache, max_tokens: int) -> str:
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=max_tokens, verbose=False, prompt_cache=cache)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--layers", default="0,4,8,12,16,20,23",
                    help="comma-separated layer ids to calibrate")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--mode", choices=["full", "fast"], default="full")
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--grad-steps", type=int, default=200)
    ap.add_argument("--n-inits", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--out", default="calibration_artifacts/local_smoke_rotors.safetensors")
    args = ap.parse_args()

    layer_ids = [int(x) for x in args.layers.split(",")]

    print(f"[smoke] loading {args.model}", flush=True)
    from mlx_lm import load
    model, tokenizer = load(args.model)

    # ── Stage 1: capture K ─────────────────────────────────────────────────
    captured = _capture_k_per_layer(model, tokenizer, DEFAULT_CALIB_TEXT, layer_ids)
    if not captured:
        print("[smoke] no K captured — abort", flush=True)
        return 1
    head_dim = next(iter(captured.values())).shape[-1]
    print(f"[smoke] head_dim={head_dim}", flush=True)

    # ── Stage 2: calibrate per-layer ───────────────────────────────────────
    rotor_state: dict[str, "torch.Tensor"] = {}
    cos_random_list: list[float] = []
    cos_refined_list: list[float] = []
    for li, K_mx in captured.items():
        t0 = time.time()
        r = _calibrate_layer(K_mx, head_dim, args.bits, args.mode,
                             args.n_seeds, args.grad_steps, args.n_inits)
        dt = time.time() - t0
        cos_random_list.append(r["cos_random"])
        cos_refined_list.append(r["cos_refined"])
        rotor_state[f"layer_{li}.q_L"] = r["q_L"]
        if args.mode == "full":
            rotor_state[f"layer_{li}.q_R"] = r["q_R"]
        lift = r["cos_refined"] - r["cos_random"]
        print(f"[calib] layer {li}: cos_random={r['cos_random']:.4f} "
              f"-> cos_refined={r['cos_refined']:.4f} "
              f"(lift {lift:+.4f}, {dt:.1f}s)", flush=True)

    avg_random = sum(cos_random_list) / len(cos_random_list)
    avg_refined = sum(cos_refined_list) / len(cos_refined_list)
    print(f"[calib] AVG cos_random={avg_random:.4f}  "
          f"cos_refined={avg_refined:.4f}  "
          f"lift={avg_refined - avg_random:+.4f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_rotors(out_path, rotor_state)
    print(f"[calib] wrote rotors -> {out_path} ({len(rotor_state)} tensors)",
          flush=True)

    # ── Stage 3: generation comparison ─────────────────────────────────────
    # We re-load the model state by re-creating caches; mlx-lm's generate()
    # reuses model weights between calls.
    from mlx_lm.models.cache import make_prompt_cache
    n_layers = len(captured)  # not the right number; use a probe instead
    probe_cache = make_prompt_cache(model)
    n_layers = len(probe_cache)
    print(f"[gen] model has {n_layers} layers total", flush=True)

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": GENERATION_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"[gen] prompt: {GENERATION_PROMPT!r}", flush=True)

    baseline_out = _generate(model, tokenizer, chat_prompt,
                             make_prompt_cache(model), args.max_tokens)
    iso_random_caches = _build_iso_caches(n_layers, head_dim,
                                           args.bits, args.mode, None)
    iso_random_out = _generate(model, tokenizer, chat_prompt,
                               iso_random_caches, args.max_tokens)
    iso_calib_caches = _build_iso_caches(n_layers, head_dim,
                                          args.bits, args.mode, out_path)
    iso_calib_out = _generate(model, tokenizer, chat_prompt,
                              iso_calib_caches, args.max_tokens)

    print("\n══════════ RESULTS ══════════")
    print(f"\n[baseline KVCache]\n{baseline_out!r}")
    print(f"\n[iso-random (uncalibrated)]\n{iso_random_out!r}")
    print(f"\n[iso-calibrated]\n{iso_calib_out!r}")
    print()

    # Coarse signal: did calibrated stay closer to baseline than random did?
    def _exact_prefix_match(a: str, b: str) -> int:
        n = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                n += 1
            else:
                break
        return n
    prefix_random = _exact_prefix_match(baseline_out, iso_random_out)
    prefix_calib = _exact_prefix_match(baseline_out, iso_calib_out)
    print(f"[diag] exact-prefix-match vs baseline: "
          f"random={prefix_random}  calibrated={prefix_calib}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
