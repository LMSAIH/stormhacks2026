# ml/tests — quantized-model regression lock

`quantized_baseline.json` pins the quantized lip-reading ONNX model: its sha256, the exact greedy texts it
produces for 15 fixed clips (first 10 LRS3 rows + first 5 `data/raw_eval` clips by name) and the metrics it
had when locked. It does not exist until a model is locked; once it does, **commit it** (the model itself
lives in the gitignored `artifacts/`). `scripts/regress_quantized.py` runs the fp32 reference and the
quantized model on identical inputs (onnxruntime CPU, greedy CTC), enforces the gates below and compares
against this file. It needs `data/lrs3_test/0000.parquet` and `data/raw_eval/` (both gitignored).

## Workflow

```bash
cd ml
# 1. quantize -> artifacts/lipread_ctc.<variant>.onnx       (scripts/quantize_onnx.py --variant <name>)
# 2. review: full fp32-vs-quantized run, 100 LRS3 + 20 raw clips, ~2-4 min; exit 1 + a table if a gate fails
uv run python scripts/regress_quantized.py --model artifacts/lipread_ctc.<variant>.onnx
# 3. lock it (runs the full evaluation again; refuses if a gate fails, --force overrides)
uv run python scripts/regress_quantized.py --model artifacts/lipread_ctc.<variant>.onnx --update-baseline
# 4. commit the lock
git add tests/quantized_baseline.json
```

Without `--model` the newest `artifacts/lipread_ctc.<variant>.onnx` is used. `--recipe "<note>"` adds a
free-text note to the baseline (the variant/desc from the `<model>.json` that quantize_onnx.py writes and a
summary of the graph's quantization ops are recorded automatically). `--reference`, `--lrs3`, `--raw`,
`--baseline`, `--out`, `--threads` override the defaults; the JSON report with per-clip rows lands in
`artifacts/regress/<model-stem>.json`.

From then on `./smoke.sh ml` runs the **quantized regression (fast)** check: the same gates and lock on the
first 5 LRS3 + 2 raw clips, in process, ~10-25 s depending on CPU load, for the model the baseline names. It
SKIPs when there is no quantized model or no baseline (`$ML_QUANT_MODEL` / `$ML_QUANT_BASELINE` override).
`uv run pytest tests` unit-tests the gate/lock logic itself on synthetic data (no models or data needed).

## Gates (per set: `lrs3` = pre-made crops, `raw` = raw face video through our crop pipeline)

| gate | limit |
|---|---|
| `size_mb` | <= 220 |
| WER delta vs fp32 | quant WER <= fp32 WER + 1.0 point; never tighter than one word error, which only matters for `--fast` (its 5 / 2 clips are ~35 / 13 words) |
| mean argmax agreement | >= 0.95 (mean over clips of the share of frames whose argmax equals fp32's) |
| quant empty, fp32 not | 0 clips |
| io contract, clips run ok | `video` f32 [1,1,T,88,88] -> `log_probs` f32 [T,5049] unchanged; every clip preprocessed, run, finite |
| max-length input (full runs only) | real clips stitched to 250 frames (the app's 10 s cap) run, give `[250,5049]`, finite: catches a too-short trimmed position table |
| lock: sha256 + texts | model sha256 equals the baseline's and every locked clip's text is identical |

## When the lock fails

onnxruntime's CPU EP is deterministic for a given model, onnxruntime version and CPU class (texts were
identical across separate processes and with 1, 4 or default threads), so a changed text is a real change.
The output prints a word-level diff and what moved: `model sha256` (re-quantized), `environment`
(onnxruntime, CPU/ISA, thread count, library versions), `model inputs` (preprocessing, video decoding or
eval data changed: each clip's input is fingerprinted). If the change is intended, review the diff and a
full run, then repeat step 3 and commit the new file; otherwise fix the cause. int8 kernels round
differently on VNNI and non-VNNI CPUs, so a locked text can flip on another machine class: re-lock there.
