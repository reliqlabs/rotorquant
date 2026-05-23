## KV cache compression with IsoQuant (optional)

This MLX quant works with [IsoQuant](https://github.com/reliqlabs/rotorquant) for a
**~3.2× smaller KV cache** at decode time with no measurable quality cost. Useful
when you're running long context on a memory-constrained Apple Silicon machine.

**TL;DR numbers** (Leanstral 4-bit, M5, 37 Lean-generation prompts × 3 seeds):

| | Strict | Soft | Decode | Cache memory |
|---|---|---|---|---|
| Default fp16 KV cache | 61.6% ± 7.1% | 85.0% ± 3.2% | 1.00× | 1.00× |
| **iso5 random KV cache** | **59.7% ± 5.8%** | 82.9% ± 2.7% | **1.15-1.22×** slower | **3.2× smaller** |

iso5 is within baseline's own seed-to-seed jitter on strict score. No calibration
pipeline needed — random rotors at 5 bits suffice for this architecture (Leanstral
uses MLA, which limits the headroom that per-rotor calibration can buy).

### Usage

```bash
pip install mlx-vlm
git clone https://github.com/reliqlabs/rotorquant
cd rotorquant && pip install -e .
```

```python
import mlx.core as mx
from mlx_vlm import load, generate
from turboquant.iso_kv_cache import IsoKVCache
from turboquant.mlx_fused_iso_attention import make_random_quaternions

model, processor = load("mvid/Leanstral-2603-MLX-4bit")
lm = model.language_model           # iso cache attaches to the LM, not the wrapper
head_dim = 128                      # Leanstral KV head_dim
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

# mlx-vlm passes `cache` straight through to the LM forward.
out = generate(model, processor, prompt="...", prompt_cache=caches, max_tokens=256)
```

### Bit-width tradeoffs

| bits | Cache memory | Strict score | Notes |
|---|---|---|---|
| 4 | 4× smaller | 54.7% ± 6.5% | ~7 pp below baseline; only choose if memory is tight |
| **5** | **3.2× smaller** | **59.7% ± 5.8%** | **Recommended.** Baseline parity within noise |
| 6 | 2.7× smaller | 65.4% ± 4.3% | Lowest seed variance; matches or beats baseline |

### Implementation notes

- The cache uses fused Metal kernels for compress and decompress; decode is ~1.2× slower than the default cache (down from ~6.5× before kernel fusion).
- Random rotors are used here for simplicity; calibrated rotors (per the original RotorQuant paper) are also supported via `load_rotors_into_cache_factory(...)`, but for MLA models they don't measurably beat random.
- Quality numbers above are on a small Lean-generation prompt set; for other domains, run your own eval before adopting.

See [the rotorquant fork](https://github.com/reliqlabs/rotorquant) for sweep CSVs,
profile logs, and the full eval harness.
