"""Export the VSR encoder + CTC head to ONNX (for onnxruntime-node / onnxruntime-web) and check it
against PyTorch.

    uv run python scripts/export_onnx.py                      # → artifacts/lipread_ctc.onnx
    uv run python scripts/export_onnx.py --model FT_v1 --check-only

Graph:  video (1, 1, T, 88, 88) float32, normalised (see preprocess.py)
        → log_probs (T, 5049) float32  — greedy CTC: argmax per frame, collapse repeats, drop 0.
Token strings: artifacts/tokens.json (index → piece; "▁" = word boundary). Beam search and the
attention decoder stay in Python (tier 3).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lipread.model import LipReader

ML = Path(__file__).resolve().parents[1]


class CtcHead(torch.nn.Module):
    def __init__(self, e2e: torch.nn.Module):
        super().__init__()
        self.encoder = e2e.encoder
        self.ctc_lo = e2e.ctc.ctc_lo

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        enc, _ = self.encoder(video, None)
        return torch.log_softmax(self.ctc_lo(enc), dim=-1)[0]


def check(path: Path, head: CtcHead, lengths=(37, 73, 151)) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for t in lengths:
        x = torch.randn(1, 1, t, 88, 88)
        with torch.no_grad():
            ref = head(x).numpy()
        t0 = time.perf_counter()
        out = sess.run(None, {"video": x.numpy()})[0]
        dt = time.perf_counter() - t0
        diff = float(np.abs(out - ref).max())
        agree = float((out.argmax(-1) == ref.argmax(-1)).mean())
        print(f"T={t:4d}: max|Δ|={diff:.2e} argmax agree={agree:.1%} ort_cpu={dt * 1000:.0f}ms")
        # fp32 drift grows with T (~8e-4 at 151 frames); argmax agreement is what decoding sees.
        if diff > 5e-3 or agree < 0.99:
            raise SystemExit(f"ONNX/PyTorch mismatch at T={t}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LRS3_V_WER19.1", help="checkpoint dir name under checkpoints/")
    ap.add_argument("--out", type=Path, default=ML / "artifacts" / "lipread_ctc.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    reader = LipReader(model_name=a.model, device="cpu", use_lm=False, beam_size=1)
    head = CtcHead(reader.e2e).eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if not a.check_only:
        t0 = time.perf_counter()
        torch.onnx.export(
            head,
            (torch.randn(1, 1, 50, 88, 88),),
            str(a.out),
            input_names=["video"],
            output_names=["log_probs"],
            dynamic_axes={"video": {2: "T"}, "log_probs": {0: "T"}},
            opset_version=a.opset,
            dynamo=False,
        )
        (a.out.parent / "tokens.json").write_text(json.dumps(reader.token_list))
        mb = a.out.stat().st_size / 1e6
        print(f"exported {a.out} ({mb:.0f} MB) in {time.perf_counter() - t0:.0f}s")
    check(a.out, head)


if __name__ == "__main__":
    main()
