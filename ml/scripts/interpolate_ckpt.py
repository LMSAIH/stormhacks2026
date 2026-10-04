"""WiSE-FT: blend fine-tuned weights with the stock ones, θ = α·θ_ft + (1−α)·θ_stock.

Fine-tuning on a narrow set (a few speakers, a fixed grammar) makes the model forget open speech;
interpolating back towards 19.1 keeps much of the in-domain gain while recovering the general
accuracy (Wortsman et al., "Robust fine-tuning of zero-shot models", 2022). No training needed:
write a few α and bench them.

    uv run python scripts/interpolate_ckpt.py FT_v1 --alphas 0.25 0.5     # → checkpoints/FT_v1_a0.25/…
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

ML = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("finetuned", help="model dir name under the checkpoint dir, e.g. FT_v1")
    ap.add_argument("--base", default="LRS3_V_WER19.1")
    ap.add_argument("--alphas", type=float, nargs="+", required=True, help="weight on the fine-tuned model")
    ap.add_argument("--ckpt-dir", type=Path, default=Path(os.environ.get("LIPREAD_CKPT_DIR", ML / "checkpoints")))
    a = ap.parse_args()

    base = torch.load(a.ckpt_dir / a.base / "model.pth", map_location="cpu", weights_only=False)
    ft = torch.load(a.ckpt_dir / a.finetuned / "model.pth", map_location="cpu", weights_only=False)
    if base.keys() != ft.keys():
        raise SystemExit("state dicts differ in keys — not the same architecture/layout")
    for alpha in a.alphas:
        out = {k: (alpha * ft[k].float() + (1 - alpha) * base[k].float()).to(base[k].dtype)
               if base[k].is_floating_point() else ft[k] for k in base}
        dst = a.ckpt_dir / f"{a.finetuned}_a{alpha:g}"
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(out, dst / "model.pth")
        shutil.copy(a.ckpt_dir / a.base / "model.json", dst / "model.json")
        print(f"α={alpha:g} → {dst}/model.pth")


if __name__ == "__main__":
    main()
