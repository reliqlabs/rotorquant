"""Calibrate IsoQuant rotors against real Leanstral K activations.

Loads the Leanstral FP8 reference via HF transformers on a Modal H200,
runs prefill on a calibration corpus, captures K cache tensors from each
attention layer, then optimizes (random-search MVP) the per-layer
quaternion rotors that minimize quantization error.

Outputs land in the `rotorquant-calibration` volume, keyed by
(model, bits, mode), as a single `.safetensors` file with q_L and q_R
per layer. The file is small (~50 KB total) — easy to download to the
M5 and feed into our MLX IsoQuant kernels.

Cost: H200 80 GB at ~$3.95/hr × ~1 hour for the streaming convert (first
run only) + ~20 min calibration. Under $5 per full run.

Run:
    modal run modal_apps/calibrate_rotorquant.py::calibrate
        --bits 3 --mode full --n-rotor-seeds 64 --calib-tokens 4096
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import modal

from modal_apps._common import (
    HF_INTERMEDIATE_DIR,
    LEANSTRAL_MODELS_PATH,
    LEANSTRAL_MODELS_VOL,
    ROTORQUANT_CALIB_PATH,
    ROTORQUANT_CALIB_VOL,
    build_image,
    prepare_hf_intermediate_if_missing,
)


app = modal.App("leanstral-calibrate", image=build_image())

# Leanstral dequantized to bf16 is ~238 GB. H100:3 (240 GB combined) is the
# smallest configuration that fits in pure VRAM with device_map="auto". H200
# alone (141 GB) requires CPU offload via accelerate, which makes calibration
# prohibitively slow.
# Going with H100:3 — predictable cost, no offload pauses.
GPU = "H100:3"
TIMEOUT_S = 4 * 60 * 60  # 4 hours max: first-run conversion + calibration
SHARED_VOLUMES = {
    LEANSTRAL_MODELS_PATH: LEANSTRAL_MODELS_VOL,
    ROTORQUANT_CALIB_PATH: ROTORQUANT_CALIB_VOL,
}


_CALIB_PROMPTS = [
    # Lean 4 proof tactics — domain-specific, our primary target.
    "import Mathlib\nimport Mathlib.Tactic\n\ntheorem add_comm_nat (a b : Nat) : a + b = b + a := by\n  induction a with\n  | zero => simp\n  | succ a ih => rw [Nat.succ_add, ih, Nat.add_succ]",
    "import Mathlib\n\ntheorem zero_le (n : Nat) : 0 ≤ n := by\n  induction n with\n  | zero => exact Nat.le_refl 0\n  | succ n ih => exact Nat.le_succ_of_le ih",
    "import Mathlib\n\ntheorem mul_zero_eq_zero (n : Nat) : n * 0 = 0 := by rfl\n\ntheorem zero_mul_eq_zero (n : Nat) : 0 * n = 0 := by\n  induction n with\n  | zero => rfl\n  | succ n ih => rw [Nat.mul_succ, ih, Nat.add_zero]",
    "import Mathlib.Data.List.Basic\n\ntheorem length_append (xs ys : List α) : (xs ++ ys).length = xs.length + ys.length := by\n  induction xs with\n  | nil => simp\n  | cons x xs ih => simp [List.cons_append, List.length_cons, ih, Nat.succ_add]",
    "import Mathlib.Topology.Basic\n\nexample {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]\n    (f : X → Y) (hf : Continuous f) (U : Set Y) (hU : IsOpen U) :\n    IsOpen (f ⁻¹' U) := hf.isOpen_preimage U hU",
    "import Mathlib.Analysis.Calculus.Deriv.Basic\n\nexample : deriv (fun x => x^2) = (fun x => 2*x) := by\n  ext x\n  simp [deriv_pow]",
    # Math prose for distributional coverage.
    "Solve the integral ∫ sin(x) cos(x) dx using the substitution u = sin(x), du = cos(x) dx, giving ∫ u du = u²/2 + C = sin²(x)/2 + C.",
    "The ε-δ definition of a limit states that limₓ→a f(x) = L iff for every ε > 0 there exists δ > 0 such that |x - a| < δ implies |f(x) - L| < ε. Continuity at a means this limit equals f(a).",
    "Cantor's diagonal argument shows that the set of reals in [0, 1] is uncountable. Assume a bijection f : ℕ → [0, 1] exists; construct a real r whose nth digit differs from f(n)'s nth digit. Then r is not in the image, contradicting surjectivity.",
    "The cardinality of the continuum c = 2^ℵ₀. The continuum hypothesis asks whether ℵ₁ = c; it is independent of ZFC.",
    # Code / general English.
    "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b",
    "Merge sort recursively splits an array of length n into halves, sorts each, then merges in O(n) time. Total complexity T(n) = 2 T(n/2) + Θ(n) = Θ(n log n). Stable and not in-place by default.",
    "A category C consists of objects and morphisms (arrows) between them, with composition and identities satisfying associativity and unit laws. A functor F : C → D maps objects to objects and morphisms to morphisms preserving composition: F(g ∘ f) = F(g) ∘ F(f).",
    # More dense math text to extend the corpus.
    "The Riemann zeta function ζ(s) = Σ_{n=1}^∞ 1/n^s converges for Re(s) > 1 and has analytic continuation to ℂ \\ {1}. Its non-trivial zeros all lie on Re(s) = 1/2 by the Riemann hypothesis.",
    "In linear algebra, the singular value decomposition factorizes any m×n matrix A as U Σ V^T where U and V are orthogonal and Σ is diagonal with non-negative singular values. Used for PCA, low-rank approximation, and pseudo-inverses.",
    "Bayes' theorem: P(A|B) = P(B|A) P(A) / P(B). Updates belief in A given new evidence B. Foundation of Bayesian inference and used throughout statistics, machine learning, and decision theory.",
]


@app.function(
    gpu=GPU,
    volumes=SHARED_VOLUMES,
    timeout=TIMEOUT_S,
    memory=200 * 1024,  # request 200 GB host memory for the HF intermediate prep
)
def calibrate(
    bits: int = 3,
    mode: str = "full",
    n_rotor_seeds: int = 64,
    calib_tokens: int = 4096,
    capture_layers: Optional[str] = None,
    output_tag: str = "default",
    optimize: str = "random",
    grad_steps: int = 300,
    grad_lr: float = 1e-2,
):
    """Capture K activations + search for low-error rotor seeds per layer.

    Args:
        bits: quantization bits (2, 3, or 4)
        mode: 'full' (q_L v q̄_R, 6 DOF) or 'fast' (q_L v, 3 DOF)
        n_rotor_seeds: how many random quaternion seeds to evaluate
        calib_tokens: how many calibration tokens to prefill
        capture_layers: comma-separated layer indices, e.g. "0,8,17,26,35"
                        (default: every 4th layer to cap memory)
        output_tag: subdir under /mnt/calibration/iso/ to write into
        optimize: 'random' (n_rotor_seeds tries), or 'gradient' (Adam-refine
            the best random seed via straight-through estimator on the argmin
            quantize). Gradient is ~30s/layer extra and typically lifts cos
            sim by 0.01-0.02.
        grad_steps: Adam iterations per layer when optimize='gradient'.
        grad_lr: Adam learning rate on rotor params.
    """
    import sys
    sys.path.insert(0, "/opt/rotorquant")

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from turboquant.isoquant import IsoQuantMSE, make_random_unit_quaternion

    print(f"[calib] bits={bits} mode={mode} seeds={n_rotor_seeds} "
          f"calib_tokens={calib_tokens}", flush=True)

    prepare_hf_intermediate_if_missing()

    print(f"[calib] loading {HF_INTERMEDIATE_DIR}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(HF_INTERMEDIATE_DIR)
    # Leanstral is mistral3 (multimodal wrapper) when vision is present.
    # AutoModelForCausalLM doesn't know about it; use the generic AutoModel
    # which routes via _model_mapping to Mistral3ForConditionalGeneration.
    cfg = AutoConfig.from_pretrained(HF_INTERMEDIATE_DIR)
    # The kernels package's grouped_mm MoE dispatch doesn't support our static
    # FP8 activation scheme. Force 'dynamic' so forward picks the eager path
    # that does work; the stored activation_scale tensors just become no-ops.
    _coerce_activation_scheme_to_dynamic(cfg)
    if cfg.model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            HF_INTERMEDIATE_DIR,
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            HF_INTERMEDIATE_DIR,
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    print(f"[calib] load() in {time.time() - t0:.1f}s", flush=True)

    # Discover head_dim per the text config (Leanstral: 128).
    text_cfg = getattr(model.config, "text_config", model.config)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // text_cfg.num_attention_heads
    )
    n_layers = text_cfg.num_hidden_layers
    print(f"[calib] head_dim={head_dim}, n_layers={n_layers}", flush=True)
    assert head_dim % 4 == 0, "IsoQuant needs head_dim divisible by 4"
    n_groups = head_dim // 4

    if capture_layers is None:
        # Default: every 4th layer, plus the last one
        layer_ids = sorted(set(list(range(0, n_layers, 4)) + [n_layers - 1]))
    else:
        layer_ids = [int(x) for x in capture_layers.split(",")]
    print(f"[calib] capturing layers: {layer_ids}", flush=True)

    # Tokenize the calibration corpus to ≤ calib_tokens.
    text = "\n\n".join(_CALIB_PROMPTS)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=calib_tokens)
    input_ids = enc.input_ids.to(next(model.parameters()).device)
    print(f"[calib] prefilling {input_ids.shape[1]} tokens", flush=True)

    # Use the model's normal `use_cache=True` path so transformers populates
    # a cache with the actual K used inside attention — no fragile per-
    # architecture hooks needed.
    t1 = time.time()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    print(f"[calib] forward pass in {time.time() - t1:.1f}s", flush=True)
    # One-shot diagnostic of layer-cache attrs to confirm we're reading the right field.
    if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
        layer0 = past_kv.layers[0]
        layer0_attrs = sorted(a for a in dir(layer0) if not a.startswith("_"))
        print(f"[calib] past_kv.layers[0] type: {type(layer0).__name__}", flush=True)
        print(f"[calib] past_kv.layers[0] attrs: {layer0_attrs}", flush=True)
        for attr in ("keys", "key_states", "key_cache", "k"):
            v = getattr(layer0, attr, None)
            if v is not None:
                print(f"[calib] layers[0].{attr}: shape={getattr(v, 'shape', None)}",
                      flush=True)
                break

    # past_kv may be a DynamicCache or a list-of-tuples depending on transformers
    # version. Normalize to a list of K tensors per layer.
    captured: dict[int, torch.Tensor] = {}
    for li in layer_ids:
        K = _extract_k_for_layer(past_kv, li)
        if K is None:
            print(f"[calib]   layer {li}: could not extract K from cache, skip",
                  flush=True)
            continue
        # K shape: (B, n_kv_heads, seq, head_dim)
        captured[li] = K.detach().reshape(-1, K.shape[-1]).to(torch.float32).cpu()
        print(f"[calib]   layer {li}: K {tuple(K.shape)} -> {captured[li].shape[0]} vectors",
              flush=True)

    # Per layer: run random-rotor search, save best q_L (+q_R).
    out_dir = f"{ROTORQUANT_CALIB_PATH}/iso/{output_tag}/bits{bits}-{mode}"
    os.makedirs(out_dir, exist_ok=True)
    summary: dict[str, dict] = {}
    rotor_state: dict[str, torch.Tensor] = {}
    # If gradient-refining, do it on GPU (K is on cpu after the model forward
    # but moving back to cuda is cheap — 31k vectors × 128 fp32 = ~16 MB).
    grad_device = None
    if optimize == "gradient":
        import torch
        grad_device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    for li in layer_ids:
        if li not in captured:
            continue
        K = captured[li]  # already (N, head_dim) fp32 on cpu
        best = _random_rotor_search(K, head_dim, bits, mode, n_rotor_seeds)
        if optimize == "gradient":
            K_gpu = K.to(grad_device)
            refined = _gradient_refine_rotors(
                K_gpu, head_dim, bits, mode, init=best,
                n_steps=grad_steps, lr=grad_lr,
            )
            if refined["cos_mean"] >= best["cos_mean"]:
                best = refined
                print(f"[calib]   layer {li}: gradient refine lift "
                      f"{best['cos_mean'] - refined['cos_mean']:+.4f} (kept)", flush=True)
        rotor_state[f"layer_{li}.q_L"] = best["q_L"]
        if mode == "full":
            rotor_state[f"layer_{li}.q_R"] = best["q_R"]
        summary[f"layer_{li}"] = {
            "seed": best["seed"],
            "cosine_mean": best["cos_mean"],
            "cosine_p05": best["cos_p05"],
            "mse": best["mse"],
            "n_vectors": K.shape[0],
        }
        print(f"[calib] layer {li}: seed={best['seed']:>5} "
              f"cos={best['cos_mean']:.4f} (p5={best['cos_p05']:.4f}) "
              f"mse={best['mse']:.4e}", flush=True)

    rotors_path = f"{out_dir}/rotors.safetensors"
    save_file(rotor_state, rotors_path, metadata={"format": "pt"})
    summary_path = f"{out_dir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "bits": bits,
                "mode": mode,
                "n_rotor_seeds": n_rotor_seeds,
                "calib_tokens": int(input_ids.shape[1]),
                "captured_layers": layer_ids,
                "head_dim": head_dim,
                "n_groups": n_groups,
            },
            "per_layer": summary,
        }, f, indent=2)
    ROTORQUANT_CALIB_VOL.commit()
    print(f"[calib] wrote {rotors_path} ({len(rotor_state)} tensors)", flush=True)
    print(f"[calib] wrote {summary_path}", flush=True)
    return {"rotors_path": rotors_path, "summary_path": summary_path, "summary": summary}


# ── Hook / search helpers (run inside the Modal container) ──────────────────


def _coerce_activation_scheme_to_dynamic(cfg):
    """Mutate cfg.quantization_config so the MoE forward picks the eager path.

    Handles both the dict and FP8Config object shapes that transformers may
    surface depending on version. Also flips dequantize=True so transformers
    expands FP8 → bf16 at load time, bypassing the kernels-community Triton
    kernels entirely — those kernels have a shape-mismatch bug with the
    current transformers version on the grouped_mm experts path.
    """
    qc = getattr(cfg, "quantization_config", None)
    if qc is None:
        return
    if hasattr(qc, "activation_scheme"):
        qc.activation_scheme = "dynamic"
        qc.dequantize = True
    elif isinstance(qc, dict):
        qc["activation_scheme"] = "dynamic"
        qc["dequantize"] = True


def _find_decoder_layers(model):
    """Return a list of transformer decoder layers (Mistral4/Mistral3 wrapper aware)."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return list(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "language_model"):
        return list(model.language_model.model.layers)
    raise RuntimeError(f"could not locate layers on model of type {type(model)}")


def _find_attention(layer):
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)


def _extract_k_for_layer(past_kv, layer_idx: int):
    """Pull layer_idx's K tensor out of whatever cache transformers gave us.

    Supports three layouts:
      - transformers 5.6+ DynamicCache: `past_kv.layers[i]` is a KVCacheLayer
        with `.keys` (preferred) or `.key_cache`.
      - transformers 5.x older DynamicCache: `past_kv.key_cache` is a list.
      - Legacy tuple-of-tuples format.
    """
    if hasattr(past_kv, "layers"):
        layers = past_kv.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            for attr in ("keys", "key_states", "key_cache", "k"):
                v = getattr(layer, attr, None)
                if v is not None:
                    return v
    if hasattr(past_kv, "key_cache"):
        kc = past_kv.key_cache
        if layer_idx < len(kc) and kc[layer_idx] is not None:
            return kc[layer_idx]
    if isinstance(past_kv, (tuple, list)) and layer_idx < len(past_kv):
        entry = past_kv[layer_idx]
        if isinstance(entry, (tuple, list)) and len(entry) >= 1:
            return entry[0]
    return None


def _random_rotor_search(K, head_dim: int, bits: int, mode: str, n_seeds: int) -> dict:
    """Try N random quaternion seeds, return the one with lowest reconstruction MSE."""
    import sys
    sys.path.insert(0, "/opt/rotorquant")
    import torch
    from turboquant.isoquant import IsoQuantMSE

    best = None
    for seed in range(n_seeds):
        iso = IsoQuantMSE(head_dim, bits, mode=mode, seed=seed, device="cpu")
        with torch.no_grad():
            x_hat, _ = iso(K)
        mse = (K - x_hat).pow(2).mean().item()
        nx = K.norm(dim=-1).clamp(min=1e-8)
        ny = x_hat.norm(dim=-1).clamp(min=1e-8)
        cos = (K * x_hat).sum(dim=-1) / (nx * ny)
        cos_mean = cos.mean().item()
        cos_p05 = cos.quantile(0.05).item()
        if best is None or mse < best["mse"]:
            best = {
                "seed": seed,
                "mse": mse,
                "cos_mean": cos_mean,
                "cos_p05": cos_p05,
                "q_L": iso.q_L.detach().clone(),
                "q_R": iso.q_R.detach().clone() if mode == "full" else None,
            }
    return best


def _quat_conj_torch(q):
    """Quaternion conjugate (w, -x, -y, -z) using torch."""
    import torch
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def _quat_mul_torch(a, b):
    """Hamilton product of two quaternion tensors (..., 4)."""
    import torch
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([rw, rx, ry, rz], dim=-1)


def _gradient_refine_rotors(K, head_dim: int, bits: int, mode: str,
                            init: dict, n_steps: int = 300, lr: float = 1e-2,
                            n_inits: int = 8) -> dict:
    """Multi-start gradient refinement via the shared turboquant.iso_calibrate
    module. The single-source-of-truth implementation lives there so it can be
    unit-tested locally (no Modal needed). See:
        turboquant/iso_calibrate.py::gradient_refine_rotors
        tests/test_iso_calibrate.py
    """
    from turboquant.iso_calibrate import gradient_refine_rotors
    result = gradient_refine_rotors(
        K, head_dim, bits, mode=mode, init=init,
        n_steps=n_steps, lr=lr, n_inits=n_inits,
        loss_kind="cos", forward_kind="soft", soft_temperature=0.05,
    )
    result["seed"] = init.get("seed", -1)
    return result


@app.local_entrypoint()
def main(
    bits: int = 3,
    mode: str = "full",
    n_rotor_seeds: int = 64,
    calib_tokens: int = 4096,
    capture_layers: str = "",
    output_tag: str = "default",
    optimize: str = "random",
    grad_steps: int = 300,
    grad_lr: float = 1e-2,
):
    result = calibrate.remote(
        bits=bits,
        mode=mode,
        n_rotor_seeds=n_rotor_seeds,
        calib_tokens=calib_tokens,
        capture_layers=capture_layers or None,
        output_tag=output_tag,
        optimize=optimize,
        grad_steps=grad_steps,
        grad_lr=grad_lr,
    )
    print("---- calibration summary ----")
    print(json.dumps(result["summary"], indent=2))
    print(f"---- rotors at: {result['rotors_path']} ----")
    print(f"---- download via: modal volume get rotorquant-calibration "
          f"{result['rotors_path'].replace('/mnt/calibration/', '')} ./rotors.safetensors")
