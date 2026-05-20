"""Run the Lean eval harness against the FP8 Leanstral reference on Modal H200.

Establishes the "high-watermark" Lean quality numbers for any quant comparison.
Uses the same prompt set + scoring as `tools/lean_eval_harness.py`, but with
torch + transformers generation against the FP8 HF intermediate (no MLX).

Output: a CSV in the calibration volume plus a printed summary. Cost ~$5/run
for the prep step (first run only) + ~5-10 min generation on H200.

Run:
    modal run modal_apps/baseline_fp8_eval.py::main --max-tokens 256
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

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


app = modal.App("leanstral-eval", image=build_image())
GPU = "H200"
TIMEOUT_S = 4 * 60 * 60
SHARED_VOLUMES = {
    LEANSTRAL_MODELS_PATH: LEANSTRAL_MODELS_VOL,
    ROTORQUANT_CALIB_PATH: ROTORQUANT_CALIB_VOL,
}


# Same 8 prompts as the local harness — keeps comparisons apples-to-apples.
PROMPTS = [
    {"name": "add_zero", "user": "Write a Lean 4 theorem proving n + 0 = n for natural numbers, using only `rfl` if possible. Output only the Lean 4 code.",
     "expects": {"theorem": True, "tactic": "rfl"}},
    {"name": "zero_add", "user": "Write a Lean 4 theorem stating 0 + n = n for natural numbers. Prove it with one tactic. Output only the Lean 4 code.",
     "expects": {"theorem": True, "tactic": "simp"}},
    {"name": "add_comm", "user": "Prove commutativity of natural number addition in Lean 4: for all a b, a + b = b + a. Use induction. Output only the Lean 4 code.",
     "expects": {"theorem": True, "tactic": "induction"}},
    {"name": "list_length", "user": "Define a Lean 4 function `length : List α → Nat` that returns the length of a list. Output only the Lean 4 code.",
     "expects": {"theorem": False, "function": True}},
    {"name": "even_double", "user": "Prove in Lean 4 that 2 * n is always even, where Even is defined as ∃ k, n = 2 * k. Output only the Lean 4 code.",
     "expects": {"theorem": True, "tactic": "exact"}},
    {"name": "succ_inj", "user": "State and prove in Lean 4 that the successor function on Nat is injective. Output only the Lean 4 code.",
     "expects": {"theorem": True}},
    {"name": "le_refl", "user": "Prove reflexivity of ≤ on natural numbers in Lean 4: for all n, n ≤ n. Output only the Lean 4 code.",
     "expects": {"theorem": True}},
    {"name": "and_comm", "user": "Prove in Lean 4 that propositional conjunction is commutative: P ∧ Q ↔ Q ∧ P. Output only the Lean 4 code.",
     "expects": {"theorem": True, "tactic": "constructor"}},
]


_THEOREM_RE = re.compile(r"\b(theorem|lemma|example)\s+\w*\s*(?:\([^)]*\)\s*)*:", re.DOTALL)
_BY_RE = re.compile(r":=\s*by\b")
_FN_RE = re.compile(r"\b(def|fun)\s+\w+")
_IMPORTS_RE = re.compile(r"^\s*import\s+\w+", re.MULTILINE)
_UNICODE_NAT = re.compile(r"\bℕ\b")


def _score(text: str, expects: dict) -> dict:
    s = {
        "has_theorem":   bool(_THEOREM_RE.search(text)),
        "has_by":        bool(_BY_RE.search(text)),
        "has_function":  bool(_FN_RE.search(text)),
        "imports":       len(_IMPORTS_RE.findall(text)),
        "unicode_nat":   bool(_UNICODE_NAT.search(text)),
        "char_count":    len(text),
        "ends_with_keyword": text.rstrip().endswith((":= rfl", ":= by", "done", "sorry")),
    }
    if "tactic" in expects:
        s[f"used_{expects['tactic']}"] = bool(
            re.search(rf"\b{re.escape(expects['tactic'])}\b", text)
        )
    pts = 0
    if expects.get("theorem") and s["has_theorem"]:
        pts += 1
    if expects.get("function") and s["has_function"]:
        pts += 1
    if "tactic" in expects and s.get(f"used_{expects['tactic']}", False):
        pts += 1
    s["score"] = pts
    return s


@app.function(
    gpu=GPU,
    volumes=SHARED_VOLUMES,
    timeout=TIMEOUT_S,
    memory=200 * 1024,
)
def run_eval(max_tokens: int = 256, output_tag: str = "fp8-baseline") -> dict:
    sys.path.insert(0, "/opt/rotorquant")

    import torch
    from transformers import AutoConfig, AutoTokenizer

    prepare_hf_intermediate_if_missing()

    print(f"[eval] loading {HF_INTERMEDIATE_DIR}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(HF_INTERMEDIATE_DIR)
    cfg = AutoConfig.from_pretrained(HF_INTERMEDIATE_DIR)
    if cfg.model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            HF_INTERMEDIATE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            HF_INTERMEDIATE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    print(f"[eval] load() in {time.time() - t0:.1f}s", flush=True)

    out_dir = Path(f"{ROTORQUANT_CALIB_PATH}/eval/{output_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"lean_eval_max{max_tokens}.csv"

    fields = [
        "name", "score", "char_count", "wall_seconds", "tok_per_sec",
        "has_theorem", "has_by", "has_function", "imports",
        "unicode_nat", "ends_with_keyword", "output",
    ]
    summary = {"per_prompt": [], "total_score": 0, "max_score": 0}

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, p in enumerate(PROMPTS):
            print(f"[eval] [{i + 1}/{len(PROMPTS)}] {p['name']}", flush=True)
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["user"]}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt = f"<s>[INST] {p['user']} [/INST]"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t1 = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy for reproducibility
                    pad_token_id=tokenizer.eos_token_id or 2,
                )
            dt = time.time() - t1
            new_ids = out_ids[0, inputs.input_ids.shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)

            scores = _score(text, p["expects"])
            row = {
                "name": p["name"],
                "score": scores["score"],
                "char_count": scores["char_count"],
                "wall_seconds": round(dt, 2),
                "tok_per_sec": round(max_tokens / dt, 3),
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

            max_expected = (
                int(bool(p["expects"].get("theorem"))) +
                int(bool(p["expects"].get("function"))) +
                int("tactic" in p["expects"])
            )
            summary["per_prompt"].append({
                "name": p["name"], "score": scores["score"], "max": max_expected,
                "tok_per_sec": round(max_tokens / dt, 2),
            })
            summary["total_score"] += scores["score"]
            summary["max_score"] += max_expected
            print(f"[eval]   score={scores['score']}/{max_expected} "
                  f"{dt:.1f}s  has_thm={scores['has_theorem']} "
                  f"has_by={scores['has_by']}", flush=True)

    pct = summary["total_score"] / summary["max_score"] * 100 if summary["max_score"] else 0
    print(f"[eval] total: {summary['total_score']}/{summary['max_score']} ({pct:.1f}%)", flush=True)
    print(f"[eval] CSV -> {csv_path}", flush=True)
    ROTORQUANT_CALIB_VOL.commit()
    summary["csv_path"] = str(csv_path)
    return summary


@app.local_entrypoint()
def main(max_tokens: int = 256, output_tag: str = "fp8-baseline"):
    summary = run_eval.remote(max_tokens=max_tokens, output_tag=output_tag)
    print("---- summary ----")
    print(json.dumps(summary, indent=2))
