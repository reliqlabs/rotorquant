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
    # ── Nat / Int / basic arithmetic ─────────────────────────────────────────
    "import Mathlib\nimport Mathlib.Tactic\n\ntheorem add_comm_nat (a b : Nat) : a + b = b + a := by\n  induction a with\n  | zero => simp\n  | succ a ih => rw [Nat.succ_add, ih, Nat.add_succ]",
    "import Mathlib\n\ntheorem zero_le (n : Nat) : 0 ≤ n := by\n  induction n with\n  | zero => exact Nat.le_refl 0\n  | succ n ih => exact Nat.le_succ_of_le ih",
    "import Mathlib\n\ntheorem mul_zero_eq_zero (n : Nat) : n * 0 = 0 := by rfl\n\ntheorem zero_mul_eq_zero (n : Nat) : 0 * n = 0 := by\n  induction n with\n  | zero => rfl\n  | succ n ih => rw [Nat.mul_succ, ih, Nat.add_zero]",
    "import Mathlib\n\ntheorem succ_add (m n : Nat) : (m + 1) + n = (m + n) + 1 := by\n  induction n with\n  | zero => rfl\n  | succ n ih => rw [Nat.add_succ, Nat.add_succ, ih]",
    "import Mathlib\n\ntheorem add_assoc_nat (a b c : Nat) : (a + b) + c = a + (b + c) := by\n  induction c with\n  | zero => rfl\n  | succ c ih => rw [Nat.add_succ, Nat.add_succ, Nat.add_succ, ih]",
    "import Mathlib\n\ntheorem mul_succ_eq (a b : Nat) : a * (b + 1) = a * b + a := Nat.mul_succ a b\n\ntheorem succ_pred_lt (n : Nat) (h : 0 < n) : n.pred < n := Nat.pred_lt (Nat.pos_iff_ne_zero.mp h)",
    "import Mathlib\n\nexample (n : Nat) : n + 0 = n := Nat.add_zero n\nexample (n : Nat) : n * 1 = n := Nat.mul_one n\nexample (n : Nat) : 1 * n = n := Nat.one_mul n",
    # ── List / induction ─────────────────────────────────────────────────────
    "import Mathlib.Data.List.Basic\n\ntheorem length_append (xs ys : List α) : (xs ++ ys).length = xs.length + ys.length := by\n  induction xs with\n  | nil => simp\n  | cons x xs ih => simp [List.cons_append, List.length_cons, ih, Nat.succ_add]",
    "import Mathlib.Data.List.Basic\n\ntheorem reverse_append (xs ys : List α) : (xs ++ ys).reverse = ys.reverse ++ xs.reverse := by\n  induction xs with\n  | nil => simp\n  | cons x xs ih => simp [List.cons_append, List.reverse_cons, ih, List.append_assoc]",
    "import Mathlib.Data.List.Basic\n\ntheorem reverse_reverse (xs : List α) : xs.reverse.reverse = xs := by\n  induction xs with\n  | nil => rfl\n  | cons x xs ih => simp [List.reverse_cons, List.reverse_append, ih]",
    "import Mathlib.Data.List.Basic\n\ntheorem map_append (f : α → β) (xs ys : List α) : (xs ++ ys).map f = xs.map f ++ ys.map f := by\n  induction xs with\n  | nil => simp\n  | cons x xs ih => simp [List.cons_append, List.map_cons, ih]",
    "import Mathlib.Data.List.Basic\n\nexample : [1, 2, 3].map (· * 2) = [2, 4, 6] := by decide\nexample : [1, 2, 3, 4].filter (· > 2) = [3, 4] := by decide\nexample : [1, 2, 3].foldr (· + ·) 0 = 6 := by decide",
    # ── Propositional logic / decidability ───────────────────────────────────
    "theorem and_comm_prop (p q : Prop) : p ∧ q ↔ q ∧ p := by\n  constructor\n  · rintro ⟨hp, hq⟩; exact ⟨hq, hp⟩\n  · rintro ⟨hq, hp⟩; exact ⟨hp, hq⟩",
    "theorem not_not_iff (p : Prop) [Decidable p] : ¬¬p ↔ p := by\n  constructor\n  · intro hnn; by_contra hp; exact hnn hp\n  · intro hp hnp; exact hnp hp",
    "theorem or_comm_prop (p q : Prop) : p ∨ q ↔ q ∨ p := by\n  constructor\n  · rintro (hp | hq); · exact Or.inr hp; · exact Or.inl hq\n  · rintro (hq | hp); · exact Or.inr hq; · exact Or.inl hp",
    "theorem demorgan (p q : Prop) : ¬(p ∨ q) ↔ ¬p ∧ ¬q := by\n  constructor\n  · intro h; exact ⟨fun hp => h (Or.inl hp), fun hq => h (Or.inr hq)⟩\n  · rintro ⟨hnp, hnq⟩ (hp | hq); · exact hnp hp; · exact hnq hq",
    "theorem imp_iff_not_or (p q : Prop) [Decidable p] : (p → q) ↔ (¬p ∨ q) := by\n  by_cases hp : p\n  · simp [hp]; intro h; exact h hp\n  · simp [hp]; intro hp'; exact absurd hp' hp",
    # ── Group theory ─────────────────────────────────────────────────────────
    "import Mathlib.Algebra.Group.Defs\n\nexample {G : Type*} [Group G] (a : G) : a * a⁻¹ = 1 := mul_inv_cancel a\nexample {G : Type*} [Group G] (a : G) : a⁻¹ * a = 1 := inv_mul_cancel a",
    "import Mathlib.Algebra.Group.Defs\n\nexample {G : Type*} [Group G] (a b : G) : (a * b)⁻¹ = b⁻¹ * a⁻¹ := by\n  rw [mul_inv_rev]\n\nexample {G : Type*} [Group G] (a : G) : (a⁻¹)⁻¹ = a := inv_inv a",
    "import Mathlib.Algebra.Group.Defs\n\nexample {G : Type*} [Group G] (a b c : G) (h : a * b = a * c) : b = c := mul_left_cancel h\nexample {G : Type*} [Group G] (a b c : G) (h : b * a = c * a) : b = c := mul_right_cancel h",
    "import Mathlib.Algebra.Group.Basic\n\nexample {G : Type*} [CommGroup G] (a b : G) : a * b = b * a := mul_comm a b\nexample {G : Type*} [Group G] : (1 : G)⁻¹ = 1 := inv_one",
    # ── Ring / field ─────────────────────────────────────────────────────────
    "import Mathlib.Algebra.Ring.Basic\n\nexample {R : Type*} [Ring R] (a : R) : a * 0 = 0 := mul_zero a\nexample {R : Type*} [Ring R] (a b c : R) : a * (b + c) = a * b + a * c := mul_add a b c",
    "import Mathlib.Algebra.Ring.Basic\n\nexample {R : Type*} [Ring R] (a b : R) : (a + b)^2 = a^2 + 2*a*b + b^2 := by ring\nexample {R : Type*} [CommRing R] (a b : R) : (a - b) * (a + b) = a^2 - b^2 := by ring",
    "import Mathlib.Algebra.Field.Basic\n\nexample {F : Type*} [Field F] (a : F) (h : a ≠ 0) : a * a⁻¹ = 1 := mul_inv_cancel₀ h\nexample {F : Type*} [Field F] (a b : F) (h : a ≠ 0) : a * b / a = b := by field_simp",
    # ── Real analysis / linarith / nlinarith ─────────────────────────────────
    "import Mathlib.Data.Real.Basic\n\nexample (x : ℝ) (hx : 0 < x) : 1 / x > 0 := by positivity\nexample (a b : ℝ) (h : a < b) : a + 1 < b + 1 := by linarith",
    "import Mathlib.Data.Real.Basic\n\nexample (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := le_trans h1 h2\nexample (a b : ℝ) (h : a < b) : a ≤ b := le_of_lt h",
    "import Mathlib.Data.Real.Basic\nimport Mathlib.Tactic.Linarith\n\nexample (a b c : ℝ) (h1 : a + b ≤ 5) (h2 : a + c ≤ 4) (h3 : b + c ≤ 6) : a + b + c ≤ 7.5 := by linarith",
    "import Mathlib.Tactic.Polyrith\n\nexample (x y : ℝ) (h : x = 2 ∧ y = 3) : x + y = 5 := by\n  obtain ⟨hx, hy⟩ := h\n  rw [hx, hy]; ring",
    # ── Calculus / differentiability ─────────────────────────────────────────
    "import Mathlib.Analysis.Calculus.Deriv.Basic\n\nexample : deriv (fun x => x^2) = (fun x => 2*x) := by\n  ext x\n  simp [deriv_pow]",
    "import Mathlib.Analysis.Calculus.Deriv.Basic\n\nexample : deriv (fun (x : ℝ) => x^3 + 2*x + 1) = (fun x => 3*x^2 + 2) := by\n  ext x\n  simp [deriv_add, deriv_pow, deriv_mul_const, deriv_const]\n  ring",
    "import Mathlib.Analysis.SpecialFunctions.Exp\n\nexample : deriv (fun x => Real.exp x) = Real.exp := by\n  ext x; exact Real.deriv_exp x",
    "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic\n\nexample : deriv Real.sin = Real.cos := by ext x; exact Real.deriv_sin x\nexample : deriv Real.cos = fun x => -Real.sin x := by ext x; exact Real.deriv_cos x",
    # ── Topology ─────────────────────────────────────────────────────────────
    "import Mathlib.Topology.Basic\n\nexample {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]\n    (f : X → Y) (hf : Continuous f) (U : Set Y) (hU : IsOpen U) :\n    IsOpen (f ⁻¹' U) := hf.isOpen_preimage U hU",
    "import Mathlib.Topology.Basic\n\nexample {X : Type*} [TopologicalSpace X] (s : Set X) :\n    IsClosed s ↔ IsOpen sᶜ := isClosed_iff_isOpen_compl",
    "import Mathlib.Topology.ContinuousFunction.Basic\n\nexample {X Y Z : Type*} [TopologicalSpace X] [TopologicalSpace Y] [TopologicalSpace Z]\n    (f : X → Y) (g : Y → Z) (hf : Continuous f) (hg : Continuous g) : Continuous (g ∘ f) :=\n  hg.comp hf",
    "import Mathlib.Topology.Connected.Basic\n\nexample {X : Type*} [TopologicalSpace X] (s : Set X) (h : IsConnected s) (h_open : IsOpen s) : IsPreconnected s := h.2",
    # ── Set theory ───────────────────────────────────────────────────────────
    "import Mathlib.Data.Set.Basic\n\nexample {α : Type*} (s t : Set α) : s ⊆ t ↔ ∀ x, x ∈ s → x ∈ t := Iff.rfl\n\nexample {α : Type*} (s : Set α) : s ∪ ∅ = s := Set.union_empty s",
    "import Mathlib.Data.Set.Basic\n\nexample {α : Type*} (s t : Set α) : s ∩ t = t ∩ s := Set.inter_comm s t\nexample {α : Type*} (s : Set α) : s ∩ s = s := Set.inter_self s",
    "import Mathlib.Data.Set.Basic\n\nexample {α : Type*} (s t u : Set α) : (s ∪ t) ∪ u = s ∪ (t ∪ u) := Set.union_assoc s t u\nexample {α : Type*} (s : Set α) : sᶜᶜ = s := compl_compl s",
    # ── Combinatorics ────────────────────────────────────────────────────────
    "import Mathlib.Combinatorics.Choose.Basic\n\nexample : Nat.choose 5 2 = 10 := by decide\nexample (n : Nat) : Nat.choose n 0 = 1 := Nat.choose_zero_right n\nexample (n : Nat) : Nat.choose n n = 1 := Nat.choose_self n",
    "import Mathlib.Combinatorics.Choose.Basic\n\nexample (n k : Nat) (h : k ≤ n) : Nat.choose n k = Nat.choose n (n - k) := Nat.choose_symm h\nexample (n : Nat) : (Finset.range (n+1)).sum (Nat.choose n) = 2^n := Nat.sum_range_choose n",
    # ── Number theory ────────────────────────────────────────────────────────
    "import Mathlib.NumberTheory.Divisors\n\nexample : Nat.gcd 12 18 = 6 := by decide\nexample (n : Nat) : Nat.gcd n n = n := Nat.gcd_self n\nexample (a b : Nat) : Nat.gcd a b = Nat.gcd b a := Nat.gcd_comm a b",
    "import Mathlib.NumberTheory.Prime.Basic\n\nexample : Nat.Prime 7 := by decide\nexample : ¬ Nat.Prime 1 := Nat.not_prime_one\nexample (p : Nat) (hp : Nat.Prime p) : p ≥ 2 := hp.two_le",
    # ── Function composition / injectivity / bijection ──────────────────────
    "import Mathlib.Logic.Function.Basic\n\nexample {α β : Type*} (f : α → β) : Function.Injective f ↔ ∀ a b, f a = f b → a = b := Iff.rfl\nexample {α : Type*} : Function.Injective (id : α → α) := fun _ _ h => h",
    "import Mathlib.Logic.Function.Basic\n\nexample {α β γ : Type*} (f : β → γ) (g : α → β)\n    (hf : Function.Injective f) (hg : Function.Injective g) :\n    Function.Injective (f ∘ g) := hf.comp hg",
    # ── Inductive types / structures ────────────────────────────────────────
    "inductive Tree (α : Type) where\n  | leaf : Tree α\n  | node : Tree α → α → Tree α → Tree α\n\ndef Tree.size : Tree α → Nat\n  | .leaf => 0\n  | .node l _ r => 1 + l.size + r.size",
    "inductive Vec (α : Type) : Nat → Type where\n  | nil : Vec α 0\n  | cons : α → Vec α n → Vec α (n+1)\n\ndef Vec.length : Vec α n → Nat\n  | _ => n",
    "structure Point (α : Type) where\n  x : α\n  y : α\n  deriving Repr\n\ndef Point.translate (p : Point Int) (dx dy : Int) : Point Int :=\n  { x := p.x + dx, y := p.y + dy }",
    # ── Type classes ────────────────────────────────────────────────────────
    "class Monoid' (α : Type) where\n  one : α\n  mul : α → α → α\n  one_mul : ∀ a, mul one a = a\n  mul_one : ∀ a, mul a one = a\n  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)",
    "class Functor' (F : Type → Type) where\n  map : (α → β) → F α → F β\n  id_map : ∀ x, map id x = x\n  comp_map : ∀ (g : β → γ) (f : α → β) x, map (g ∘ f) x = map g (map f x)",
    # ── Equation compiler / dependent pattern matching ──────────────────────
    "def Nat.factorial : Nat → Nat\n  | 0 => 1\n  | n + 1 => (n + 1) * Nat.factorial n\n\nexample : Nat.factorial 5 = 120 := by decide",
    "def fib : Nat → Nat\n  | 0 => 0\n  | 1 => 1\n  | n + 2 => fib n + fib (n + 1)\n\nexample : fib 10 = 55 := by decide",
    # ── Misc tactic showcase ────────────────────────────────────────────────
    "import Mathlib.Tactic\n\nexample (n : Nat) (h : n = 5) : n + 3 = 8 := by omega\nexample (a b : Int) (h1 : a + b = 10) (h2 : a - b = 4) : a = 7 ∧ b = 3 := by omega",
    "import Mathlib.Tactic\n\nexample (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = a + b) : c = 5 := by\n  subst h3; rw [h1, h2]; norm_num",
    "import Mathlib.Tactic\n\nexample : ∃ n : Nat, n > 100 := ⟨101, by decide⟩\nexample : ∀ n : Nat, n + 1 > n := fun n => Nat.lt_succ_self n",
    "import Mathlib.Tactic\n\nexample {α : Type*} [DecidableEq α] (a b : α) (h : a = b) : decide (a = b) = true := by simp [h]\nexample (n : Nat) (h : 0 < n) : n - 1 + 1 = n := Nat.sub_add_cancel h",
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
    per_head: bool = False,
    calibrate_v: bool = False,
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
          f"calib_tokens={calib_tokens} per_head={per_head} "
          f"calibrate_v={calibrate_v} optimize={optimize}",
          flush=True)

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
    # version. Normalize to {layer_idx: {"k": tensor, "v": tensor}} per layer.
    # When per_head=False (legacy), we flatten heads into rows: (N, head_dim).
    # When per_head=True, we keep heads separate: (n_heads, N_per_head, head_dim).
    # When calibrate_v=False, "v" is missing and we only calibrate K (legacy).
    captured: dict[int, dict[str, torch.Tensor]] = {}
    n_kv_heads_observed = None
    for li in layer_ids:
        K, V = _extract_kv_for_layer(past_kv, li)
        if K is None:
            print(f"[calib]   layer {li}: could not extract K from cache, skip",
                  flush=True)
            continue
        # K, V shape: (B, n_kv_heads, seq, head_dim)
        K_t = K.detach().to(torch.float32).cpu()
        n_kv_heads_observed = K_t.shape[1]
        entry: dict[str, torch.Tensor] = {}
        if per_head:
            entry["k"] = K_t.permute(1, 0, 2, 3).reshape(K_t.shape[1], -1, K_t.shape[-1])
        else:
            entry["k"] = K_t.reshape(-1, K_t.shape[-1])
        if calibrate_v and V is not None:
            V_t = V.detach().to(torch.float32).cpu()
            if per_head:
                entry["v"] = V_t.permute(1, 0, 2, 3).reshape(
                    V_t.shape[1], -1, V_t.shape[-1])
            else:
                entry["v"] = V_t.reshape(-1, V_t.shape[-1])
        captured[li] = entry
        kshape = tuple(entry["k"].shape)
        vshape = tuple(entry["v"].shape) if "v" in entry else None
        print(f"[calib]   layer {li}: K shape={kshape}"
              + (f", V shape={vshape}" if vshape else ""), flush=True)

    # Per layer (× per head if requested): random-rotor search → save rotors.
    out_dir = f"{ROTORQUANT_CALIB_PATH}/iso/{output_tag}/bits{bits}-{mode}"
    os.makedirs(out_dir, exist_ok=True)
    summary: dict[str, dict] = {}
    rotor_state: dict[str, torch.Tensor] = {}
    grad_device = None
    if optimize == "gradient":
        import torch
        grad_device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    def _run_one(K_block, label: str) -> dict:
        """Random search (+ optional gradient refine) on a single K block."""
        best = _random_rotor_search(K_block, head_dim, bits, mode, n_rotor_seeds)
        if optimize == "gradient":
            K_gpu = K_block.to(grad_device)
            refined = _gradient_refine_rotors(
                K_gpu, head_dim, bits, mode, init=best,
                n_steps=grad_steps, lr=grad_lr,
            )
            if refined["cos_mean"] >= best["cos_mean"]:
                lift = refined["cos_mean"] - best["cos_mean"]
                best = refined
                print(f"[calib]   {label}: gradient refine lift "
                      f"{lift:+.4f} (kept)", flush=True)
        return best

    def _save_kind(li: int, kv_kind: str, best: dict, prefix_head: str = "") -> None:
        """Insert best rotor into rotor_state under the right key.
        kv_kind: 'k', 'v', or '' (single shared rotor — legacy).
        prefix_head: '' for per-layer, '_head_{hi}' for per-head.
        """
        kv_prefix = (kv_kind + "_") if kv_kind else ""
        stem = f"layer_{li}{prefix_head}"
        rotor_state[f"{stem}.{kv_prefix}q_L"] = best["q_L"]
        if mode == "full":
            rotor_state[f"{stem}.{kv_prefix}q_R"] = best["q_R"]

    for li in layer_ids:
        entry = captured.get(li)
        if entry is None:
            continue
        # If V is present we'll save both K and V rotors under .k_q_L / .v_q_L
        # so the loader picks up the K/V split. Otherwise stick to the legacy
        # single-rotor format (.q_L).
        has_v = "v" in entry
        kinds: list[str] = ["k", "v"] if has_v else [""]

        if per_head:
            K_layer = entry["k"]  # (H, N_per_head, head_dim)
            n_heads = K_layer.shape[0]
            layer_summary: dict = {"n_heads": n_heads,
                                    "n_vectors_per_head": K_layer.shape[1]}
            for kind in kinds:
                X = entry[kind if kind else "k"]
                cos_h, p05_h = [], []
                for hi in range(n_heads):
                    best = _run_one(X[hi],
                                    f"layer {li} head {hi}"
                                    + (f" [{kind.upper()}]" if kind else ""))
                    _save_kind(li, kind, best, prefix_head=f"_head_{hi}")
                    cos_h.append(best["cos_mean"])
                    p05_h.append(best["cos_p05"])
                tag = kind if kind else "shared"
                layer_summary[f"{tag}_cosine_mean_avg"] = sum(cos_h) / n_heads
                layer_summary[f"{tag}_cosine_mean_min"] = min(cos_h)
                layer_summary[f"{tag}_cosine_mean_max"] = max(cos_h)
                layer_summary[f"{tag}_cosine_p05_avg"] = sum(p05_h) / n_heads
                print(f"[calib] layer {li}{(' [' + kind.upper() + ']') if kind else ''}: "
                      f"H={n_heads} cos avg={sum(cos_h) / n_heads:.4f} "
                      f"(min={min(cos_h):.4f} max={max(cos_h):.4f})", flush=True)
            summary[f"layer_{li}"] = layer_summary
        else:
            layer_summary = {}
            for kind in kinds:
                X = entry[kind if kind else "k"]
                best = _run_one(X, f"layer {li}" + (f" [{kind.upper()}]" if kind else ""))
                _save_kind(li, kind, best)
                tag = kind if kind else "shared"
                layer_summary[f"{tag}_seed"] = best["seed"]
                layer_summary[f"{tag}_cosine_mean"] = best["cos_mean"]
                layer_summary[f"{tag}_cosine_p05"] = best["cos_p05"]
                layer_summary[f"{tag}_mse"] = best["mse"]
                layer_summary[f"{tag}_n_vectors"] = X.shape[0]
                print(f"[calib] layer {li}{(' [' + kind.upper() + ']') if kind else ''}: "
                      f"seed={best['seed']:>5} cos={best['cos_mean']:.4f} "
                      f"(p5={best['cos_p05']:.4f}) mse={best['mse']:.4e}",
                      flush=True)
            summary[f"layer_{li}"] = layer_summary

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


def _extract_kv_for_layer(past_kv, layer_idx: int):
    """Pull (K, V) tensors for `layer_idx` out of the transformers cache.

    Supports three layouts:
      - transformers 5.6+ DynamicCache: `past_kv.layers[i]` has `.keys` and
        `.values` (preferred), or `.key_cache`/`.value_cache`.
      - older DynamicCache: `past_kv.key_cache` / `.value_cache` are lists.
      - legacy tuple-of-tuples format: `past_kv[i] = (K, V)`.

    Returns (K, V) or (None, None) if extraction fails.
    """
    if hasattr(past_kv, "layers"):
        layers = past_kv.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            K = None
            for attr in ("keys", "key_states", "key_cache", "k"):
                K = getattr(layer, attr, None)
                if K is not None:
                    break
            V = None
            for attr in ("values", "value_states", "value_cache", "v"):
                V = getattr(layer, attr, None)
                if V is not None:
                    break
            return K, V
    if hasattr(past_kv, "key_cache"):
        kc = past_kv.key_cache
        vc = getattr(past_kv, "value_cache", None)
        if layer_idx < len(kc) and kc[layer_idx] is not None:
            K = kc[layer_idx]
            V = vc[layer_idx] if vc is not None and layer_idx < len(vc) else None
            return K, V
    if isinstance(past_kv, (tuple, list)) and layer_idx < len(past_kv):
        entry = past_kv[layer_idx]
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            return entry[0], entry[1]
    return None, None


def _extract_k_for_layer(past_kv, layer_idx: int):
    """Back-compat shim that returns only K."""
    K, _ = _extract_kv_for_layer(past_kv, layer_idx)
    return K


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
    per_head: bool = False,
    calibrate_v: bool = False,
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
        per_head=per_head,
        calibrate_v=calibrate_v,
    )
    print("---- calibration summary ----")
    print(json.dumps(result["summary"], indent=2))
    print(f"---- rotors at: {result['rotors_path']} ----")
    print(f"---- download via: modal volume get rotorquant-calibration "
          f"{result['rotors_path'].replace('/mnt/calibration/', '')} ./rotors.safetensors")
