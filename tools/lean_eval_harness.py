"""Lean-flavored evaluation harness for mlx-vlm / mlx-lm models.

Loads a model, runs N Lean-themed prompts, scores each output via cheap regex
heuristics (theorem header, `:= by` tactic block, `rfl`/`simp`/`exact` usage,
imports), writes a CSV with full generations + scores. Mirrors what we'd want
to run against the Leanstral quants on the 128 GB M5.

For development on this 24 GB Mac, drop in a small mlx-lm model like
mlx-community/Qwen2.5-0.5B-Instruct-4bit to validate the harness wiring without
needing Leanstral resident in RAM.

Usage:
    python tools/lean_eval_harness.py <model_repo_or_path> [out.csv] \
        [--max-tokens 128] [--mlx-vlm] [--n-prompts 8]

Defaults: 8 built-in Lean prompts, 128 max tokens, mlx-lm loader.
Pass --mlx-vlm for vision-language models (Leanstral) that need the
Pixtral processor.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import mlx.core as mx


PROMPTS = [
    # ── Nat / arithmetic (basics) ─────────────────────────────────────
    {
        "name": "add_zero",
        "user": "Write a Lean 4 theorem proving n + 0 = n for natural numbers, using only `rfl` if possible. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "rfl"},
    },
    {
        "name": "zero_add",
        "user": "Write a Lean 4 theorem stating 0 + n = n for natural numbers. Prove it with one tactic. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "simp"},
    },
    {
        "name": "add_comm",
        "user": "Prove commutativity of natural number addition in Lean 4: for all a b, a + b = b + a. Use induction. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "induction"},
    },
    {
        "name": "add_assoc",
        "user": "Prove associativity of natural number addition in Lean 4: for all a b c, (a + b) + c = a + (b + c). Use induction. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "induction"},
    },
    {
        "name": "mul_one",
        "user": "Write a Lean 4 theorem stating n * 1 = n for natural numbers. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "one_mul",
        "user": "Prove in Lean 4 that 1 * n = n for natural numbers. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "simp"},
    },
    {
        "name": "mul_zero",
        "user": "Write a Lean 4 theorem proving n * 0 = 0 for natural numbers, with the shortest possible proof. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "rfl"},
    },
    {
        "name": "mul_comm",
        "user": "Prove commutativity of multiplication on Nat in Lean 4: a * b = b * a. Use Mathlib if needed. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    # ── Nat ordering / inequalities ────────────────────────────────────
    {
        "name": "le_refl",
        "user": "Prove reflexivity of ≤ on natural numbers in Lean 4: for all n, n ≤ n. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "le_trans",
        "user": "Prove transitivity of ≤ on Nat in Lean 4: a ≤ b and b ≤ c implies a ≤ c. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "zero_le",
        "user": "Prove in Lean 4 that 0 ≤ n for any natural number n. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "lt_succ_self",
        "user": "Prove in Lean 4 that n < n + 1 for any Nat. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    # ── Nat parity / evenness ──────────────────────────────────────────
    {
        "name": "even_double",
        "user": "Prove in Lean 4 that 2 * n is always even, where Even is defined as ∃ k, n = 2 * k. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "exact"},
    },
    {
        "name": "succ_inj",
        "user": "State and prove in Lean 4 that the successor function on Nat is injective. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    # ── Lists ──────────────────────────────────────────────────────────
    {
        "name": "list_length",
        "user": "Define a Lean 4 function `length : List α → Nat` that returns the length of a list. Output only the Lean 4 code.",
        "expects": {"theorem": False, "function": True},
    },
    {
        "name": "list_append_length",
        "user": "Prove in Lean 4 that (xs ++ ys).length = xs.length + ys.length for any two lists. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "induction"},
    },
    {
        "name": "list_reverse_reverse",
        "user": "Prove in Lean 4 that reversing a list twice gives the original list: xs.reverse.reverse = xs. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "induction"},
    },
    {
        "name": "list_map_append",
        "user": "Prove in Lean 4 that mapping a function over an appended list distributes: (xs ++ ys).map f = xs.map f ++ ys.map f. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "induction"},
    },
    {
        "name": "list_filter_def",
        "user": "Define a Lean 4 function `filter (p : α → Bool) (xs : List α) : List α` that keeps only the elements satisfying p. Output only the Lean 4 code.",
        "expects": {"theorem": False, "function": True},
    },
    # ── Propositional logic ────────────────────────────────────────────
    {
        "name": "and_comm",
        "user": "Prove in Lean 4 that propositional conjunction is commutative: P ∧ Q ↔ Q ∧ P. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "constructor"},
    },
    {
        "name": "or_comm",
        "user": "Prove in Lean 4 that propositional disjunction is commutative: P ∨ Q ↔ Q ∨ P. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "double_negation",
        "user": "Prove in Lean 4 that ¬¬P ↔ P for any decidable proposition P. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "demorgan_or",
        "user": "Prove De Morgan's law in Lean 4: ¬(P ∨ Q) ↔ ¬P ∧ ¬Q. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "constructor"},
    },
    {
        "name": "implication_chain",
        "user": "Prove in Lean 4: if P → Q and Q → R then P → R. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "intro"},
    },
    # ── Functions / injectivity ────────────────────────────────────────
    {
        "name": "id_injective",
        "user": "Prove in Lean 4 that the identity function on any type is injective. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "comp_injective",
        "user": "Prove in Lean 4 that the composition of two injective functions is injective. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    # ── Algebra (Group / Ring / Field) ─────────────────────────────────
    {
        "name": "group_mul_inv",
        "user": "Prove in Lean 4 (using Mathlib) that in a group, a * a⁻¹ = 1. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "group_inv_inv",
        "user": "Prove in Lean 4 (using Mathlib) that in a group, (a⁻¹)⁻¹ = a. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "ring_mul_zero",
        "user": "Prove in Lean 4 (using Mathlib) that in any ring, a * 0 = 0. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "ring_distrib",
        "user": "Prove in Lean 4 (using Mathlib) the left distributive law: a * (b + c) = a * b + a * c in any ring. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "binomial_sq",
        "user": "Prove in Lean 4 (using Mathlib's `ring` tactic) that (a + b)^2 = a^2 + 2*a*b + b^2 over the reals. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "ring"},
    },
    {
        "name": "diff_of_squares",
        "user": "Prove in Lean 4 (using Mathlib's `ring` tactic) that (x - y) * (x + y) = x^2 - y^2 over the reals. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "ring"},
    },
    # ── Combinatorics / Number theory ──────────────────────────────────
    {
        "name": "choose_zero",
        "user": "Prove in Lean 4 (using Mathlib) that Nat.choose n 0 = 1 for any n. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "gcd_comm",
        "user": "Prove in Lean 4 (using Mathlib) that Nat.gcd is commutative: gcd a b = gcd b a. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "gcd_self",
        "user": "Prove in Lean 4 (using Mathlib) that Nat.gcd n n = n. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    # ── Calculus / Analysis ────────────────────────────────────────────
    {
        "name": "deriv_x_squared",
        "user": "In Lean 4 with Mathlib, show that the derivative of fun x => x^2 is fun x => 2*x. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "linarith_simple",
        "user": "Prove in Lean 4 (using Mathlib's linarith tactic) that if a < b for real numbers, then a + 1 < b + 1. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "linarith"},
    },
]


# ── Output scoring (cheap regex heuristics — no Lean toolchain required) ───


# Strict — "this looks like a real theorem header".
_THEOREM_RE = re.compile(r"\b(theorem|lemma|example)\s+\w*\s*(?:\([^)]*\)\s*)*:", re.DOTALL)
_BY_RE = re.compile(r":=\s*by\b")
_FN_RE = re.compile(r"\b(def|fun)\s+\w+")
_IMPORTS_RE = re.compile(r"^\s*import\s+\w+", re.MULTILINE)
_UNICODE_NAT = re.compile(r"\bℕ\b")

# Soft — "does this output show ANY sign of being Lean 4 code at all"?
# Useful for catching outputs that are valid Lean syntax but not a textbook
# `theorem name (args) : ... := ...` block. Any of these landing is at least
# "the model knows the language."
_LEAN_KEYWORD_RE = re.compile(
    r"\b(theorem|lemma|example|def|fun|import\s+Mathlib|"
    r"induction|simp|rfl|exact|intro|apply|cases|constructor|"
    r"Nat\.|List\.|Prop\b|Type\s*[\d]*\b|⟨|⟩|≤|∀|∃|→|↦)"
)
_LEAN_CODEBLOCK_RE = re.compile(r"```\s*lean(?:4)?\b", re.IGNORECASE)


def score_output(text: str, expects: dict) -> dict:
    """Cheap heuristic scoring. Returns a flat dict of booleans + ints + a
    soft 0-3 partial-credit score so we can distinguish "complete garbage"
    from "valid Lean but no full theorem" outputs."""
    s = {
        "has_theorem":   bool(_THEOREM_RE.search(text)),
        "has_by":        bool(_BY_RE.search(text)),
        "has_function":  bool(_FN_RE.search(text)),
        "imports":       len(_IMPORTS_RE.findall(text)),
        "unicode_nat":   bool(_UNICODE_NAT.search(text)),
        "char_count":    len(text),
        "ends_with_keyword": text.rstrip().endswith((":= rfl", ":= by", "done", "sorry")),
        # New soft signals.
        "has_lean_keyword": bool(_LEAN_KEYWORD_RE.search(text)),
        "has_lean_codeblock": bool(_LEAN_CODEBLOCK_RE.search(text)),
        "lean_keyword_count": len(_LEAN_KEYWORD_RE.findall(text)),
    }
    if "tactic" in expects:
        s[f"used_{expects['tactic']}"] = bool(re.search(rf"\b{re.escape(expects['tactic'])}\b", text))

    # Strict score: original 0/2 or 0/3 — exact matches on the textbook form.
    pts = 0
    if expects.get("theorem") and s["has_theorem"]:
        pts += 1
    if expects.get("function") and s["has_function"]:
        pts += 1
    if "tactic" in expects and s.get(f"used_{expects['tactic']}", False):
        pts += 1
    s["score"] = pts

    # Soft score: partial credit on a 0-3 scale.
    soft = 0
    if s["has_lean_keyword"]:
        soft += 1
    if s["has_theorem"] or s["has_function"]:
        soft += 1
    if s["has_by"] or s["ends_with_keyword"]:
        soft += 1
    s["soft_score"] = soft
    return s


# ── Generation backends ────────────────────────────────────────────────────


def _wrap_inst(user_msg: str) -> str:
    """Manual Mistral [INST] wrap — covers models where the chat template
    didn't propagate (e.g., our early Leanstral quants)."""
    return f"<s>[INST] {user_msg} [/INST]"


def generate_mlx_lm(model, tokenizer, user_msg: str, max_tokens: int,
                    temperature: float = 1.0) -> tuple[str, float]:
    """Use mlx-lm.generate. Returns (text, wall_seconds).

    Leanstral's README recommends temperature=1.0; we default to that rather
    than greedy. Greedy on this model produced repetitive/short outputs in
    our first eval pass.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt or prompt.strip() == user_msg.strip():
            # apply_chat_template silently returned the raw user message
            # (tokenizer ships with empty .chat_template attribute). Wrap.
            prompt = _wrap_inst(user_msg)
    except Exception:
        prompt = _wrap_inst(user_msg)

    sampler = make_sampler(temp=temperature)
    t0 = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                    verbose=False, sampler=sampler)
    return text, time.time() - t0


def generate_mlx_vlm(model, processor, user_msg: str, max_tokens: int,
                     temperature: float = 1.0) -> tuple[str, float]:
    """Use mlx-vlm.generate. Returns (text, wall_seconds)."""
    import contextlib
    import mlx_vlm.generate  # noqa: F401
    _mvgen_mod = sys.modules["mlx_vlm.generate"]
    if getattr(_mvgen_mod.wired_limit, "__name__", "") != "_noop":
        @contextlib.contextmanager
        def _noop(*a, **kw):
            yield
        _mvgen_mod.wired_limit = _noop

    from mlx_vlm import generate

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    try:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt or prompt.strip() == user_msg.strip():
            prompt = _wrap_inst(user_msg)
    except Exception:
        prompt = _wrap_inst(user_msg)

    t0 = time.time()
    result = generate(model, processor, prompt=prompt, max_tokens=max_tokens,
                      verbose=False, temperature=temperature)
    text = getattr(result, "text", str(result))
    return text, time.time() - t0


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF repo or local path")
    ap.add_argument("out", nargs="?", default="lean_eval.csv", help="output CSV path")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--mlx-vlm", action="store_true",
                    help="use mlx-vlm loader (for Leanstral / Pixtral models)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature (Leanstral default 1.0; "
                         "set to 0 for greedy)")
    ap.add_argument("--print-first", action="store_true",
                    help="print the first prompt's output verbatim (debug)")
    args = ap.parse_args()

    print(f"[eval] loading {args.model} (mlx-vlm={args.mlx_vlm})", flush=True)
    if args.mlx_vlm:
        from mlx_vlm import load as mlx_load
        model, processor_or_tok = mlx_load(args.model, lazy=True)
    else:
        from mlx_lm import load as mlx_load
        model, processor_or_tok = mlx_load(args.model)

    out_path = Path(args.out)
    fields = [
        "name", "score", "soft_score", "char_count", "wall_seconds", "tok_per_sec",
        "has_theorem", "has_by", "has_function", "has_lean_keyword",
        "has_lean_codeblock", "lean_keyword_count", "imports",
        "unicode_nat", "ends_with_keyword", "output",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total_score = 0
        max_score = 0
        total_soft = 0
        for i, p in enumerate(PROMPTS[: args.n_prompts]):
            print(f"[eval] [{i + 1}/{args.n_prompts}] {p['name']}", flush=True)
            try:
                gen_fn = generate_mlx_vlm if args.mlx_vlm else generate_mlx_lm
                text, dt = gen_fn(model, processor_or_tok, p["user"], args.max_tokens,
                                  temperature=args.temperature)
            except Exception as e:
                print(f"[eval]   ERROR: {type(e).__name__}: {e}", flush=True)
                continue
            scores = score_output(text, p["expects"])
            if args.print_first and i == 0:
                print(f"[eval]   ---- raw output (first prompt) ----")
                print(text)
                print(f"[eval]   ---- end ----", flush=True)
            row = {
                "name": p["name"],
                "score": scores["score"],
                "soft_score": scores["soft_score"],
                "char_count": scores["char_count"],
                "wall_seconds": round(dt, 2),
                "tok_per_sec": round(args.max_tokens / dt, 3),
                "has_theorem": scores["has_theorem"],
                "has_by": scores["has_by"],
                "has_function": scores["has_function"],
                "has_lean_keyword": scores["has_lean_keyword"],
                "has_lean_codeblock": scores["has_lean_codeblock"],
                "lean_keyword_count": scores["lean_keyword_count"],
                "imports": scores["imports"],
                "unicode_nat": scores["unicode_nat"],
                "ends_with_keyword": scores["ends_with_keyword"],
                "output": text.replace("\n", "\\n"),
            }
            w.writerow(row)
            f.flush()
            total_score += scores["score"]
            total_soft += scores["soft_score"]
            max_expected = (
                int(bool(p["expects"].get("theorem"))) +
                int(bool(p["expects"].get("function"))) +
                int("tactic" in p["expects"])
            )
            max_score += max_expected
            print(f"[eval]   strict={scores['score']}/{max_expected}  "
                  f"soft={scores['soft_score']}/3  {dt:.1f}s  "
                  f"thm={scores['has_theorem']} by={scores['has_by']} "
                  f"any_lean={scores['has_lean_keyword']}", flush=True)

    pct = (total_score / max_score * 100) if max_score else 0
    soft_pct = (total_soft / (3 * args.n_prompts) * 100) if args.n_prompts else 0
    print(f"[eval] strict total: {total_score}/{max_score} ({pct:.1f}%)", flush=True)
    print(f"[eval] soft total:   {total_soft}/{3 * args.n_prompts} ({soft_pct:.1f}%)", flush=True)
    print(f"[eval] CSV -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
