# Training loop: from user fixes to a new model

How corrected clips and team recordings could become a new model in the app. The pipeline is
built and has run end to end on a GPU pod (Phase 1, on LRS3 test clips) and on a public rehearsal
(GRID), but Phase 2 on the team's own recordings has not run (`.context/b2-report.md`), so the app
still ships the stock `LRS3_V_WER19.1` weights. A model ships only if it passes both gates.

```mermaid
flowchart TD
  subgraph app["Browser app"]
    fix["User picks another reading or types the words"]
    opt{"Share corrected clips?<br/>opt-in, off by default"}
  end
  fix --> opt
  opt -->|"on: POST /training-pairs<br/>96×96 mouth crops, text, source"| srv["ML server<br/>saves id.npz + id.txt + id.json<br/>in LIPREAD_PAIRS_DIR"]
  srv -.->|"LIPREAD_PAIRS_REPO set:<br/>background upload"| pairsds[("HF dataset: training pairs<br/>public (D61)")]
  rec["Team recordings<br/>scripts p1–p4, 1 s of still lips around each line"] --> recds[("HF dataset: recordings<br/>private (D75)")]

  subgraph b2["B2 pod, secure RTX 4090: ml/runpod/b2_finetune.sh PHASE=2"]
    prep["prepare_finetune_data.py<br/>our MouthCropper at 25 fps, lossless .npy crops<br/>one speaker held out as the test set<br/>pairs: train split only, typed + picked"]
    ft["finetune.py, starting from 19.1<br/>frozen BatchNorm, lr 1e-4, 3 epochs, bf16<br/>best checkpoint by greedy val WER<br/>converted back with a logit check"]
    wise["interpolate_ckpt.py: WiSE-FT blends with 19.1<br/>α 0.25, 0.35, 0.4, 0.5"]
    bench["bench.py: stock vs each blend<br/>LRS3-100 and the held-out speaker, greedy + beam"]
    gate{"Ship gates (D74)<br/>held-out greedy WER at least 3 points better<br/>and LRS3-100 greedy WER ≤ 30.6%"}
  end
  pairsds --> prep
  recds --> prep
  prep --> ft --> wise --> bench --> gate
  gate -->|"fail"| stock["Keep stock LRS3_V_WER19.1<br/>(what ships today)"]

  subgraph ship["Ship path, on the laptop (needs ml/data/raw_eval)"]
    exp["export_onnx.py<br/>diff logits against PyTorch"]
    quant["quantize_onnx.py --variant dyn-pw8-rn16"]
    reg["regress_quantized.py: every gate<br/>then --update-baseline to re-lock"]
    up["Upload the int8 model + tokens.json<br/>to the HF model repo, on a new branch"]
    pin["Pin that commit in modelSpec.ts"]
  end
  gate -->|"pass"| exp
  exp --> quant --> reg --> up --> pin
  pin --> appread["App: on-device reads<br/>(Instant, Normal, drafts)"]
  gate -->|"pass"| serve["Serving pod: weights in checkpoints/<br/>LIPREAD_MODEL set to the blend<br/>(Quality reads)"]

  classDef notrun fill:#fff8e1,stroke:#f9ab00,color:#000
  class prep,ft,wise,bench,gate notrun
```

The yellow steps have run on LRS3 clips (Phase 1) and on GRID, never on team recordings.

## What the rehearsal showed (GRID, `.context/b2-report.md`)

| Model | LRS3-100 greedy WER | GRID held-out greedy WER |
|---|---|---|
| Stock 19.1 | 28.6% | 73.3% |
| Plain fine-tune, lr 1e-4, 5 epochs | 70.8% | 10.2% |
| Frozen BatchNorm, lr 1e-4, 3 epochs | 38.0% | 9.0% |
| Frozen BatchNorm, then WiSE-FT α 0.4 | 29.7% | 32.7% |

Plain fine-tuning forgot open speech (LRS3-100 went from 28.6% to 70.8% wrong). Keeping 19.1's
BatchNorm statistics and blending back towards the stock weights kept most of the gain on unseen
GRID speakers for +1.1 points on LRS3-100. GRID has a fixed six-word grammar, so its gain is mostly
vocabulary; expect much less on real speech, and GRID-tuned weights never ship (D79). α was chosen
by looking at LRS3-100 itself, so the +1.1 is slightly optimistic (D74).

## Source of truth

- `frontend/src/lib/lipreading/trainingPairs.ts`
- `frontend/src/hooks/useLipReader.ts`
- `frontend/src/components/app/lip-mode-menu.tsx`
- `frontend/src/lib/lipreading/modelSpec.ts`
- `ml/src/lipread/serve/app.py`
- `ml/src/lipread/model.py`
- `ml/runpod/b2_finetune.sh`
- `ml/runpod/bootstrap.sh`
- `ml/scripts/prepare_finetune_data.py`
- `ml/scripts/finetune.py`
- `ml/scripts/interpolate_ckpt.py`
- `ml/scripts/convert_ckpt.py`
- `ml/scripts/bench.py`
- `ml/scripts/export_onnx.py`
- `ml/scripts/quantize_onnx.py`
- `ml/scripts/regress_quantized.py`
- `ml/scripts/publish_frontend_model.sh`
- `ml/tests/quantized_baseline.json`
