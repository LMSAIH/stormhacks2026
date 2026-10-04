"""Convert VSR weights between the Chaplin/LRS3_V_WER19.1 layout (served by `lipread`) and the
auto_avsr training layout. Same architecture; only module names differ:

    chaplin  encoder.frontend.*   <->  auto_avsr  frontend.*
    chaplin  encoder.embed.0.*    <->  auto_avsr  proj_encoder.*
    everything else (encoder.encoders.*, decoder.*, ctc.*) is identical.

    # fine-tune from the 19.1 weights (better than auto_avsr's own 20.3% zoo checkpoint)
    uv run python scripts/convert_ckpt.py to-auto-avsr checkpoints/LRS3_V_WER19.1/model.pth \
        checkpoints/LRS3_V_WER19.1/auto_avsr_init.pth --verify
    # serve a fine-tuned model with `lipread` (writes model.pth + copies model.json)
    uv run python scripts/convert_ckpt.py to-chaplin exp/ft/model_avg_5.pth checkpoints/FT_v1/model.pth

--verify runs both implementations on the same random clip and compares CTC log-probs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

ML = Path(__file__).resolve().parents[1]
AUTO_AVSR = ML / "third_party" / "auto_avsr"
VOCAB = 5049
RENAMES = [("encoder.frontend.", "frontend."), ("encoder.embed.0.", "proj_encoder.")]


def load_state_dict(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:  # Lightning .ckpt
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # ModelModule wraps the E2E as `self.model`
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    return sd


def rename(sd: dict, to_auto_avsr: bool) -> dict:
    out = {}
    for k, v in sd.items():
        for chaplin, aa in RENAMES:
            src, dst = (chaplin, aa) if to_auto_avsr else (aa, chaplin)
            if k.startswith(src):
                k = dst + k[len(src):]
                break
        out[k] = v
    return out


def _purge_espnet() -> None:
    for name in [m for m in sys.modules if m == "espnet" or m.startswith("espnet.")]:
        del sys.modules[name]


def chaplin_logits(sd: dict, x: torch.Tensor) -> torch.Tensor:
    import argparse as ap
    import json

    _purge_espnet()
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

    conf = json.load(open(ML / "checkpoints" / "LRS3_V_WER19.1" / "model.json"))
    args = ap.Namespace(**(conf if isinstance(conf, dict) else conf[2]))
    m = E2E(VOCAB, args)
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        enc, _ = m.encoder(x, None)  # x: (B, C, T, H, W)
        return torch.log_softmax(m.ctc.ctc_lo(enc), -1)


def auto_avsr_logits(sd: dict, x: torch.Tensor) -> torch.Tensor:
    _purge_espnet()
    sys.path.insert(0, str(AUTO_AVSR))
    try:
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    finally:
        sys.path.remove(str(AUTO_AVSR))
    m = E2E(VOCAB, "video")
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        h = m.proj_encoder(m.frontend(x.transpose(1, 2)))  # auto_avsr takes (B, T, C, H, W)
        h, _ = m.encoder(h, None)
        return torch.log_softmax(m.ctc.ctc_lo(h), -1)


def verify(chaplin_sd: dict, aa_sd: dict) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 40, 88, 88)
    a = chaplin_logits(chaplin_sd, x)
    b = auto_avsr_logits(aa_sd, x)
    diff = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"verify: max|Δlogp|={diff:.2e} argmax agreement={agree:.1%}")
    if diff > 1e-3:
        raise SystemExit("verify FAILED: implementations disagree (check rel-pos encoding variant)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direction", choices=["to-auto-avsr", "to-chaplin"])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    sd = load_state_dict(a.src)
    out = rename(sd, to_auto_avsr=a.direction == "to-auto-avsr")
    a.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.dst)
    print(f"wrote {a.dst} ({len(out)} tensors)")
    if a.direction == "to-chaplin":
        conf = ML / "checkpoints" / "LRS3_V_WER19.1" / "model.json"
        if not (a.dst.parent / "model.json").exists():
            shutil.copy(conf, a.dst.parent / "model.json")
            print(f"copied {conf.name} → {a.dst.parent}")
    if a.verify:
        chaplin_sd, aa_sd = (sd, out) if a.direction == "to-auto-avsr" else (out, sd)
        verify(chaplin_sd, aa_sd)


if __name__ == "__main__":
    main()
