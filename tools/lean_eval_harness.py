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
        "name": "list_length",
        "user": "Define a Lean 4 function `length : List α → Nat` that returns the length of a list. Output only the Lean 4 code.",
        "expects": {"theorem": False, "function": True},
    },
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
    {
        "name": "le_refl",
        "user": "Prove reflexivity of ≤ on natural numbers in Lean 4: for all n, n ≤ n. Output only the Lean 4 code.",
        "expects": {"theorem": True},
    },
    {
        "name": "and_comm",
        "user": "Prove in Lean 4 that propositional conjunction is commutative: P ∧ Q ↔ Q ∧ P. Output only the Lean 4 code.",
        "expects": {"theorem": True, "tactic": "constructor"},
    },
]


# ── Output scoring (cheap regex heuristics — no Lean toolchain required) ───


_THEOREM_RE = re.compile(r"\b(theorem|lemma|example)\s+\w*\s*(?:\([^)]*\)\s*)*:", re.DOTALL)
_BY_RE = re.compile(r":=\s*by\b")
_FN_RE = re.compile(r"\b(def|fun)\s+\w+")
_IMPORTS_RE = re.compile(r"^\s*import\s+\w+", re.MULTILINE)
_UNICODE_NAT = re.compile(r"\bℕ\b")


def score_output(text: str, expects: dict) -> dict:
    """Cheap heuristic scoring. Returns a flat dict of booleans + ints."""
    s = {
        "has_theorem":   bool(_THEOREM_RE.search(text)),
        "has_by":        bool(_BY_RE.search(text)),
        "has_function":  bool(_FN_RE.search(text)),
        "imports":       len(_IMPORTS_RE.findall(text)),
        "unicode_nat":   bool(_UNICODE_NAT.search(text)),
        "char_count":    len(text),
        "ends_with_keyword": text.rstrip().endswith((":= rfl", ":= by", "done", "sorry")),
    }
    # Tactic hit if the expected tactic appears anywhere in the body
    if "tactic" in expects:
        s[f"used_{expects['tactic']}"] = bool(re.search(rf"\b{re.escape(expects['tactic'])}\b", text))
    # Aggregate score: 1 point for theorem/function presence + 1 for `by` if expected
    pts = 0
    if expects.get("theorem") and s["has_theorem"]:
        pts += 1
    if expects.get("function") and s["has_function"]:
        pts += 1
    if "tactic" in expects and s.get(f"used_{expects['tactic']}", False):
        pts += 1
    s["score"] = pts
    return s


# ── Generation backends ────────────────────────────────────────────────────


def _wrap_inst(user_msg: str) -> str:
    """Manual Mistral [INST] wrap — covers models where the chat template
    didn't propagate (e.g., our early Leanstral quants)."""
    return f"<s>[INST] {user_msg} [/INST]"


def generate_mlx_lm(model, tokenizer, user_msg: str, max_tokens: int) -> tuple[str, float]:
    """Use mlx-lm.generate. Returns (text, wall_seconds)."""
    from mlx_lm import generate
    # Try the tokenizer's chat template first; fall back to manual [INST].
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = _wrap_inst(user_msg)

    t0 = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return text, time.time() - t0


def generate_mlx_vlm(model, processor, user_msg: str, max_tokens: int) -> tuple[str, float]:
    """Use mlx-vlm.generate. Returns (text, wall_seconds)."""
    # Same wired_limit workaround as the smoke tests, in case CPU mode is used.
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
    except Exception:
        prompt = _wrap_inst(user_msg)

    t0 = time.time()
    result = generate(model, processor, prompt=prompt, max_tokens=max_tokens, verbose=False)
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
        "name", "score", "char_count", "wall_seconds", "tok_per_sec",
        "has_theorem", "has_by", "has_function", "imports",
        "unicode_nat", "ends_with_keyword", "output",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total_score = 0
        max_score = 0
        for i, p in enumerate(PROMPTS[: args.n_prompts]):
            print(f"[eval] [{i + 1}/{args.n_prompts}] {p['name']}", flush=True)
            try:
                gen_fn = generate_mlx_vlm if args.mlx_vlm else generate_mlx_lm
                text, dt = gen_fn(model, processor_or_tok, p["user"], args.max_tokens)
            except Exception as e:
                print(f"[eval]   ERROR: {type(e).__name__}: {e}", flush=True)
                continue
            scores = score_output(text, p["expects"])
            row = {
                "name": p["name"],
                "score": scores["score"],
                "char_count": scores["char_count"],
                "wall_seconds": round(dt, 2),
                "tok_per_sec": round(args.max_tokens / dt, 3),
                "has_theorem": scores["has_theorem"],
                "has_by": scores["has_by"],
                "has_function": scores["has_function"],
                "imports": scores["imports"],
                "unicode_nat": scores["unicode_nat"],
                "ends_with_keyword": scores["ends_with_keyword"],
                "output": text.replace("\n", "\\n"),
            }
            w.writerow(row)
            f.flush()
            total_score += scores["score"]
            max_expected = (
                int(bool(p["expects"].get("theorem"))) +
                int(bool(p["expects"].get("function"))) +
                int("tactic" in p["expects"])
            )
            max_score += max_expected
            print(f"[eval]   score={scores['score']}/{max_expected}  "
                  f"{dt:.1f}s  has_thm={scores['has_theorem']}  "
                  f"has_by={scores['has_by']}", flush=True)

    pct = (total_score / max_score * 100) if max_score else 0
    print(f"[eval] total: {total_score}/{max_score} ({pct:.1f}%)", flush=True)
    print(f"[eval] CSV -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
