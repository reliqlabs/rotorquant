## KV cache compression with IsoQuant (prototype)

This MLX quant is compatible with [IsoQuant](https://github.com/reliqlabs/rotorquant)
for a smaller KV cache at decode time. This section reports preliminary results from
a working prototype — not a production recommendation. If you want a smaller KV
cache and are comfortable evaluating quality on your own workload, the snippet
below shows how to plug it in.

### What we observed

On Leanstral-2603-MLX-4bit, with a 5-bit iso cache and random rotors, on a small
Lean-generation benchmark (37 prompts × 3 random seeds, M5, max_tokens=256,
temperature=1.0, scoring via regex heuristics — not Lean type-checking):

| | strict (mean ± stdev) | soft (mean ± stdev) | decode | cache memory |
|---|---|---|---|---|
| Default fp16 KV cache | 61.6% ± 7.1% | 85.0% ± 3.2% | 1.00× | 1.00× |
| iso, 5 bits, random rotors | 59.7% ± 5.8% | 82.9% ± 2.7% | ~1.15-1.22× slower | ~3.2× smaller |

iso-5 sat within baseline's seed-to-seed variance on this benchmark. The sample is
small (37 prompts, one narrow domain) and the scoring is text-pattern matching
rather than the Lean toolchain, so don't read too much into the headline numbers.
Re-run against your own workload before adopting.

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

- We also tested 4-bit and 6-bit iso caches and calibrated (rather than random)
  rotors. On this model the 4-bit configuration scored noticeably below baseline
  on our benchmark, and calibrated rotors did not measurably outperform random.
  Leanstral uses MLA (multi-head latent attention), where K is expanded from a
  low-rank latent — calibration may pay off more on architectures with more
  per-head variance; we have not tested those.
- Speed numbers are after fused compress/decompress Metal kernels and a bf16
  dtype-return fix. Pre-fusion the slowdown was closer to 6×.
- Quality scoring uses output-text regex heuristics. Lean type-checking might
  give a materially different picture.

See [the rotorquant fork](https://github.com/reliqlabs/rotorquant) for sweep CSVs,
profile logs, and the full eval harness.
