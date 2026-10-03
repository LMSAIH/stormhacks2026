"""Fixtures for the lip-reading engine tests (frontend/src/lib/lipreading/*.test.ts).

    uv run --directory ml python ../frontend/src/lib/lipreading/__fixtures__/engine/make_fixture.py

Writes next to this file:
  ctc_cases.json  synthetic log-prob matrices + what Python's greedy decode returns for them
                  (`LipReader.greedy` itself, run on a stub) — pins ctc.ts to lipread.model. Committed.
  crops.bin       T*96*96 uint8 pre-made mouth crops of one LRS3 test clip (HF mattymchen/lrs3-test).
  expected.json   PyTorch greedy result for crops.bin {text, confidence, ids, ...} plus the ORT-python
                  result on the exported ONNX graph. These two are LRS3-derived (no redistribution)
                  and gitignored; the golden test (LIPREAD_ONNX_TEST=1) needs them + the 775 MB model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lipread.model import LipReader, collapse_ctc, ids_to_text

HERE = Path(__file__).resolve().parent
ML = HERE.parents[5] / "ml"
PARQUET = ML / "data" / "lrs3_test" / "0000.parquet"
ONNX = ML / "artifacts" / "lipread_ctc.onnx"

# Tiny vocab with the same layout as tokens.json: <blank> first, <unk>, pieces, <eos> last.
CTC_TOKENS = ["<blank>", "<unk>", "'", "▁A", "B", "▁C", "D▁", "▁", "<eos>"]


class _LogpStub:
    """Lets us call the real `LipReader.greedy` on a given log-prob matrix."""

    def __init__(self, logp: torch.Tensor, tokens: list[str]):
        self._logp, self.token_list = logp, tokens

    def ctc_log_probs(self, _x: object) -> torch.Tensor:
        return self._logp


def python_greedy(logp: torch.Tensor, tokens: list[str]) -> tuple[str, float | None]:
    tr = LipReader.greedy(_LogpStub(logp, tokens), None)  # type: ignore[arg-type]
    return tr.text, tr.confidence


def onehot_logits(ids: list[int], vocab: int, peak: float = 6.0) -> np.ndarray:
    logits = np.zeros((len(ids), vocab), dtype=np.float32)
    logits[np.arange(len(ids)), ids] = peak
    return logits


def ctc_cases(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    v = len(CTC_TOKENS)
    eos, unk = v - 1, 1
    named: list[tuple[str, np.ndarray]] = [
        ("empty", np.zeros((0, v), dtype=np.float32)),
        ("all blank", onehot_logits([0, 0, 0], v)),
        ("repeats merge", onehot_logits([3, 3, 4, 4, 4], v)),
        ("blank splits a repeat", onehot_logits([3, 0, 3, 4], v)),
        ("eos dropped after merge", onehot_logits([3, eos, 3, eos, eos], v)),
        ("eos only", onehot_logits([eos, eos], v)),
        ("unk kept", onehot_logits([unk, 0, 5], v)),
        ("word marker trims", onehot_logits([7, 3, 0, 6, 7, 7, 0], v)),
        ("apostrophe", onehot_logits([5, 2, 4, 0, 3], v)),
    ]
    tie = np.zeros((4, v), dtype=np.float32)
    tie[0, [4, 6]] = 3.0  # equal maxima: first index wins
    tie[1, [0, 5]] = 3.0
    tie[2, [5, 6]] = 3.0
    tie[3, :] = 1.0  # all equal → blank
    named.append(("ties pick first index", tie))
    for i, t in enumerate([1, 2, 5, 9, 16, 16, 24, 24, 40, 40]):
        logits = rng.normal(0.0, 2.5, size=(t, v)).astype(np.float32)
        logits[:, 0] += 2.0 if i % 2 else 0.0  # CTC-like blank bias on half the cases
        named.append((f"random T={t} #{i}", logits))

    out = []
    for name, logits in named:
        logp = torch.log_softmax(torch.from_numpy(logits), dim=-1)
        text, conf = python_greedy(logp, CTC_TOKENS)
        out.append({
            "name": name,
            "timesteps": int(logp.shape[0]),
            # exact float32 values (as doubles) → bit-identical once in a Float32Array
            "log_probs": logp.flatten().tolist(),
            "ids": logp.argmax(dim=-1).tolist(),
            "text": text,
            "confidence": conf,
        })
    return out


def golden(row: int) -> tuple[np.ndarray, dict]:
    import onnxruntime as ort
    import pyarrow.parquet as pq

    from lipread.preprocess import precropped_patches, to_model_input

    t = pq.ParquetFile(PARQUET).read_row_group(0, columns=["idx", "video", "label"]).slice(row, 1)
    crops = np.asarray(t.column("video")[0].as_py(), dtype=np.uint8)
    patches = precropped_patches(crops)  # (T, 96, 96) uint8
    x = to_model_input(patches)  # (1, T, 88, 88)

    reader = LipReader(device="cpu", use_lm=False, beam_size=1)
    logp = reader.ctc_log_probs(x).cpu()
    tr = reader.greedy(x)
    ids = logp.argmax(dim=-1).tolist()

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    ort_logp = sess.run(None, {"video": x.unsqueeze(0).numpy()})[0]
    ort_ids = ort_logp.argmax(-1).tolist()
    expected = {
        "source": f"{PARQUET.name} row {row} (idx {t.column('idx')[0].as_py()})",
        "label": t.column("label")[0].as_py().strip(),
        "frames": int(patches.shape[0]),
        "height": int(patches.shape[1]),
        "width": int(patches.shape[2]),
        "text": tr.text,
        "confidence": tr.confidence,
        "ids": ids,
        "onnx_text": ids_to_text(collapse_ctc(ort_ids), reader.token_list),
        "onnx_argmax_agree": float(np.mean(np.asarray(ort_ids) == np.asarray(ids))),
        "onnx_max_abs_diff": float(np.abs(ort_logp - logp.numpy()).max()),
    }
    return patches, expected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=0, help="row of the LRS3 parquet shard")
    ap.add_argument("--skip-golden", action="store_true", help="only write ctc_cases.json")
    a = ap.parse_args()

    cases = ctc_cases()
    (HERE / "ctc_cases.json").write_text(json.dumps({"tokens": CTC_TOKENS, "cases": cases}) + "\n")
    print(f"ctc_cases.json: {len(cases)} cases")

    if a.skip_golden:
        return
    patches, expected = golden(a.row)
    (HERE / "crops.bin").write_bytes(np.ascontiguousarray(patches).tobytes())
    (HERE / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"crops.bin: {patches.shape}; text={expected['text']!r} (label {expected['label']!r}); "
          f"onnx={expected['onnx_text']!r} agree={expected['onnx_argmax_agree']:.1%}")


if __name__ == "__main__":
    main()
