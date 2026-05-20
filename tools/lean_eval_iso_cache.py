"""End-to-end Lean eval with IsoQuant KV cache wired into the model forward.

Mirrors `tools/lean_eval_harness.py` but the cache is our `IsoKVCache`,
optionally loaded from a calibrated rotors.safetensors. Both modes (regular
and iso) can run side by side so you get the quality-vs-baseline delta in
the same CSV.

This is the final M5-side tool that uses every piece we built today:
    MLX quant on HF -> mlx-lm load -> IsoKVCache(calibrated rotors) ->
    iso_compress on every K, V update -> attention via decompressed cache
    -> Lean output -> regex score.

Usage:
    # Baseline cache:
    python tools/lean_eval_iso_cache.py mvid/Leanstral-2603-MLX-4bit \\
        --mode baseline --out baseline.csv

    # Iso cache from random rotors (no calibration):
    python tools/lean_eval_iso_cache.py mvid/Leanstral-2603-MLX-4bit \\
        --mode iso-random --bits 3 --iso-mode full --out iso-rand.csv

    # Iso cache from calibrated rotors:
    python tools/lean_eval_iso_cache.py mvid/Leanstral-2603-MLX-4bit \\
        --mode iso-calibrated \\
        --rotors calibration_artifacts/leanstral-iso3-full-full36-rotors.safetensors \\
        --bits 3 --iso-mode full --out iso-cal.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, ".")

# Re-use the eval harness's score + prompts so the CSVs are directly comparable.
from tools.lean_eval_harness import PROMPTS, score_output, _wrap_inst  # noqa: E402
from turboquant.iso_kv_cache import IsoKVCache, load_rotors_into_cache_factory  # noqa: E402
from turboquant.mlx_fused_iso_attention import make_random_quaternions  # noqa: E402


def _build_iso_caches(n_layers: int, head_dim: int, bits: int, iso_mode: str,
                       rotors_path: str | None) -> list:
    """Build a list of cache objects, one per layer.

    If `rotors_path` is given, use the calibrated rotors where available;
    layers without calibrated rotors fall back to random quaternions.
    """
    if rotors_path:
        factory = load_rotors_into_cache_factory(rotors_path, head_dim, bits)
    else:
        factory = lambda li: None  # noqa: E731

    fallback_q_L = make_random_quaternions(head_dim // 4, seed=1)
    fallback_q_R = make_random_quaternions(head_dim // 4, seed=2) if iso_mode == "full" else None

    out = []
    n_calibrated = 0
    for li in range(n_layers):
        c = factory(li)
        if c is None:
            c = IsoKVCache(bits=bits, q_L=fallback_q_L,
                           q_R=fallback_q_R, head_dim=head_dim)
        else:
            n_calibrated += 1
        out.append(c)
    print(f"[iso] built {n_layers} caches "
          f"({n_calibrated} from rotors, {n_layers - n_calibrated} random fallback)",
          flush=True)
    return out


def _infer_head_dim(cache_list) -> int:
    """Find the head_dim from the baseline cache list."""
    for c in cache_list:
        for attr in ("keys", "head_dim"):
            v = getattr(c, attr, None)
            if v is not None:
                if hasattr(v, "shape") and v.ndim == 4:
                    return v.shape[-1]
                if isinstance(v, int):
                    return v
    return 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["baseline", "iso-random", "iso-calibrated"],
                    default="baseline")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--iso-mode", choices=["full", "fast"], default="full")
    ap.add_argument("--rotors", default=None,
                    help="path to rotors.safetensors (only for iso-calibrated)")
    ap.add_argument("--out", default="lean_eval_iso.csv")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--print-first", action="store_true")
    args = ap.parse_args()

    if args.mode == "iso-calibrated" and not args.rotors:
        ap.error("--rotors is required when --mode=iso-calibrated")

    print(f"[iso-eval] loading {args.model} (mode={args.mode})", flush=True)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)

    # Probe head_dim + n_layers by running a 1-token forward with the
    # default cache — robust across mlx-lm model classes that don't
    # consistently expose head_dim as an attribute.
    from mlx_lm.models.cache import make_prompt_cache
    probe_cache = make_prompt_cache(model)
    probe_tokens = mx.array(tokenizer.encode("x"))[None, :]
    _ = model(probe_tokens, cache=probe_cache)
    mx.eval(_)
    head_dim = None
    for c in probe_cache:
        K = getattr(c, "keys", None)
        if K is None and hasattr(c, "state"):
            K = c.state[0]
        if K is not None and K.ndim == 4:
            head_dim = K.shape[-1]
            break
    if head_dim is None:
        head_dim = 128  # fallback to Leanstral default
    n_layers = len(probe_cache)
    print(f"[iso-eval] probed head_dim={head_dim} n_layers={n_layers}", flush=True)
    if args.mode == "baseline":
        from mlx_lm.models.cache import make_prompt_cache
        # Note: we re-create a fresh baseline cache per prompt below.
        build_cache = lambda: make_prompt_cache(model)
    elif args.mode == "iso-random":
        build_cache = lambda: _build_iso_caches(n_layers, head_dim,
                                                args.bits, args.iso_mode, None)
    else:  # iso-calibrated
        build_cache = lambda: _build_iso_caches(n_layers, head_dim,
                                                args.bits, args.iso_mode, args.rotors)

    sampler = make_sampler(temp=args.temperature) if args.temperature > 0 else None

    out_path = Path(args.out)
    fields = [
        "name", "score", "soft_score", "char_count", "wall_seconds", "tok_per_sec",
        "has_theorem", "has_by", "has_function", "has_lean_keyword",
        "has_lean_codeblock", "lean_keyword_count", "imports",
        "unicode_nat", "ends_with_keyword", "output",
    ]
    total_score = 0
    total_soft = 0
    max_score = 0
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, p in enumerate(PROMPTS[: args.n_prompts]):
            print(f"[iso-eval] [{i + 1}/{args.n_prompts}] {p['name']}", flush=True)
            cache = build_cache()
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["user"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                if not prompt or prompt.strip() == p["user"].strip():
                    prompt = _wrap_inst(p["user"])
            except Exception:
                prompt = _wrap_inst(p["user"])
            t0 = time.time()
            try:
                kwargs = dict(max_tokens=args.max_tokens, verbose=False, prompt_cache=cache)
                if sampler is not None:
                    kwargs["sampler"] = sampler
                text = generate(model, tokenizer, prompt=prompt, **kwargs)
            except Exception as e:
                print(f"[iso-eval]   ERROR: {type(e).__name__}: {e}", flush=True)
                continue
            dt = time.time() - t0
            scores = score_output(text, p["expects"])
            if args.print_first and i == 0:
                print(f"[iso-eval]   ---- raw output ----")
                print(text)
                print(f"[iso-eval]   ---- end ----", flush=True)
            row = {
                "name": p["name"],
                "score": scores["score"], "soft_score": scores["soft_score"],
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
            print(f"[iso-eval]   strict={scores['score']}/{max_expected}  "
                  f"soft={scores['soft_score']}/3  {dt:.1f}s  "
                  f"thm={scores['has_theorem']} any_lean={scores['has_lean_keyword']}",
                  flush=True)

    pct = (total_score / max_score * 100) if max_score else 0
    soft_pct = (total_soft / (3 * args.n_prompts) * 100) if args.n_prompts else 0
    print(f"[iso-eval] mode={args.mode}", flush=True)
    print(f"[iso-eval] strict {total_score}/{max_score} ({pct:.1f}%)", flush=True)
    print(f"[iso-eval] soft   {total_soft}/{3 * args.n_prompts} ({soft_pct:.1f}%)", flush=True)
    print(f"[iso-eval] CSV -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
