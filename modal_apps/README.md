# Modal apps

Cloud GPU jobs for Leanstral RotorQuant work. Run from a Modal-authenticated shell.

## Apps

### `calibrate_rotorquant.py`
Loads the FP8 Leanstral reference on a Modal H200 (141 GB VRAM), runs prefill on
a 10-prompt Lean-flavored calibration corpus, captures per-layer K tensors from
the model's DynamicCache, then searches a few dozen random quaternion seeds per
layer to find the one with the lowest IsoQuant reconstruction MSE.

Outputs a `rotors.safetensors` containing `q_L` (and `q_R` for `mode=full`) for
each calibrated layer, plus a `summary.json` with cosine sim / MSE per layer.

```
modal run modal_apps/calibrate_rotorquant.py::main \
    --bits 3 --mode full --n-rotor-seeds 64 --calib-tokens 4096
```

Download the rotors to the M5:
```
modal volume get rotorquant-calibration iso/default/bits3-full/rotors.safetensors ./rotors.safetensors
```

### `baseline_fp8_eval.py`
Runs the same 8-prompt Lean eval suite as `tools/lean_eval_harness.py` but
against the unquantized FP8 reference via torch + transformers. Establishes the
"FP8 ceiling" any MLX quant should be compared against.

```
modal run modal_apps/baseline_fp8_eval.py::main --max-tokens 256
```

Output CSV ends up at `rotorquant-calibration:eval/fp8-baseline/lean_eval_max256.csv`.

## First-run setup

Both apps share `_common.py:prepare_hf_intermediate_if_missing()`. On the first
ever call, it:
1. Downloads `mistralai/Leanstral-2603` (consolidated format) into the
   `leanstral-models` volume — ~115 GB, ~20 min.
2. Runs the streaming converter (`_convert_streaming.py`, bundled here) to
   produce the HF-format FP8 intermediate — ~30 min.

Subsequent calls reuse the prepared model in <30 s.

If `mistralai/Leanstral-2603` is gated for your account, attach an HF token via
Modal secrets:
```
modal secret create huggingface HF_TOKEN=hf_...
```
and add `secrets=[modal.Secret.from_name("huggingface")]` to the function
decorators in both apps.

## Cost guardrails

* H200 80 GB at ~$3.95/hr; the first-run prep step is ~$3, calibration is
  ~$1-2, eval is ~$1.
* `timeout=4h` per function call so a hung job can't burn unbounded budget.
* Each invocation is a fresh container — no idle billing.
* See `modal app list` to confirm nothing is sitting deployed.

## Pre-flight checklist

```
modal profile list                # confirm correct workspace
modal volume list                 # leanstral-models + rotorquant-calibration
modal run modal_apps/baseline_fp8_eval.py::main --max-tokens 32   # smoke
```
