## KV cache compression with IsoQuant (experimental)

This MLX quant is compatible with [IsoQuant](https://github.com/reliqlabs/rotorquant)
for a smaller KV cache at decode time. In our limited tests on Leanstral-2603-MLX-4bit
the 5-bit iso cache used about 3× less memory than fp16 KV while producing similar
output on a small Lean-generation benchmark. Decode was roughly 1.2× slower than the
default cache after Metal kernel fusion.

**Test setup**: 37 Lean-prompt eval × 3 random seeds on an M5, max_tokens=256,
temperature=1.0. Scoring uses regex heuristics for the presence of theorem syntax,
named tactics, etc. — not Lean type-checking. This is a thin signal on a narrow
domain; broader benchmarks may show different tradeoffs.

| | strict (mean ± stdev) | soft (mean ± stdev) | decode | cache memory |
|---|---|---|---|---|
| Default fp16 KV cache | 61.6% ± 7.1% | 85.0% ± 3.2% | 1.00× | 1.00× |
| iso, 5 bits, random rotors | 59.7% ± 5.8% | 82.9% ± 2.7% | ~1.15-1.22× slower | ~3.2× smaller |
| iso, 4 bits, random rotors | 54.7% ± 6.5% | 82.9% ± 2.4% | ~1.2× slower | ~4× smaller |
| iso, 6 bits, random rotors | 65.4% ± 4.3% | 85.0% ± 2.6% | ~1.2× slower | ~2.7× smaller |

iso-5 sits within baseline's seed-to-seed variance on this benchmark — that's the
basis for considering it a near-no-cost option, but the sample is small (37 prompts,
one domain) and we make no broader quality claims.

### Usage

```bash
pip install mlx-vlm
git clone https://github.com/reliqlabs/rotorquant
cd rotorquant && pip install -e .
```

```python
from mlx_vlm import load, generate
from turboquant.iso_kv_cache import IsoKVCache
from turboquant.mlx_fused_iso_attention import make_random_quaternions

model, processor = load("mvid/Leanstral-2603-MLX-4bit")
head_dim = 128         # Leanstral KV head_dim
n_layers = 36
n_groups = head_dim // 4

caches = [
    IsoKVCache(
        bits=5,
        q_L=make_random_quaternions(n_groups, seed=i),
        q_R=make_random_quaternions(n_groups, seed=i + 1000),
        head_dim=head_dim,
    )
    for i in range(n_layers)
]

out = generate(model, processor, prompt="...", prompt_cache=caches, max_tokens=256)
```

### Notes and caveats

- We also experimented with calibrated rotors (per the RotorQuant paper). On this
  model (Leanstral uses MLA — multi-head latent attention — where K is expanded
  from a low-rank latent) they did not measurably outperform random rotors on our
  benchmark. Calibration may still pay off on architectures with more per-head
  variance; we have not tested those.
- Per-head rotors (one per KV head per layer) were tested and did not help on this
  model for the same MLA-related reason.
- Decode speed numbers are after fused compress/decompress Metal kernels and a
  bf16 dtype-return fix. Pre-fusion the slowdown was closer to 6× — verify on
  your own hardware.
- Quality scoring uses output-text regex heuristics, not Lean type-checking. A
  more rigorous eval (e.g., Lean compiler verification) might show different
  results.

See [the rotorquant fork](https://github.com/reliqlabs/rotorquant) for sweep CSVs,
profile logs, and the full eval harness.
