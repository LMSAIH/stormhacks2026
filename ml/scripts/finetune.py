"""Fine-tune LRS3_V_WER19.1 on prepared clips (B2) with the vendored auto_avsr recipe.

Wraps auto_avsr's `ModelModule` + `DataModule` instead of its `train.py`, which assumes SLURM,
wandb and multi-node DDP and keeps checkpoints by step count. Here: one device, CSV logs, greedy
CTC val WER every epoch (the stock module only logs val loss), best-k checkpoints by val WER,
then average them, keep whichever of {best, average} has the lower val WER, convert it back to
the served (Chaplin) layout with a logit check, and score stock vs fine-tuned on val and test.

    uv run python scripts/finetune.py --root data/ft --name FT_v1 [--epochs 15 --lr 1e-4]
    LIPREAD_MODEL=FT_v1 uv run lipread transcribe clip.mp4      # use the result

Input: `scripts/prepare_finetune_data.py` output (ROOT/labels/{train,val,test}.csv).
Output: exp/<name>/ (checkpoints, metrics.csv, summary.json) and checkpoints/<name>/model.pth.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

ML = Path(__file__).resolve().parents[1]
AUTO_AVSR = ML / "third_party" / "auto_avsr"
STOCK = "LRS3_V_WER19.1"


def _import_auto_avsr():
    """auto_avsr ships its own `espnet` package; it must win over the installed Chaplin one."""
    for name in [m for m in sys.modules if m == "espnet" or m.startswith("espnet.")]:
        del sys.modules[name]
    sys.path.insert(0, str(AUTO_AVSR))
    sys.path.insert(0, str(ML / "scripts"))
    import convert_ckpt  # noqa: F401  (scripts/convert_ckpt.py)
    from datamodule.data_module import DataModule
    from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
    from lightning import ModelModule

    return ModelModule, DataModule, make_non_pad_mask


ModelModule, DataModule, make_non_pad_mask = _import_auto_avsr()
import convert_ckpt  # noqa: E402
from pytorch_lightning import Trainer, seed_everything  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger  # noqa: E402


def collapse(ids: list[int]) -> list[int]:
    out, prev = [], 0
    for t in ids:
        if t != prev and t != 0:
            out.append(t)
        prev = t
    return out


def word_errors(ref: str, hyp: str) -> int:
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)]


class WarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    """auto_avsr's WarmupCosineScheduler as a LambdaLR (theirs passes `verbose`, gone in torch 2.8)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def f(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))))

        super().__init__(optimizer, f)


class FineTuneModule(ModelModule):
    """auto_avsr ModelModule + greedy CTC WER on val, optional frozen 3D/ResNet front-end."""

    def __init__(self, args):
        super().__init__(args)
        if args.freeze_frontend:
            for p in self.model.frontend.parameters():
                p.requires_grad = False
        self._errs = self._words = 0
        self.samples: list[tuple[str, str]] = []

    def on_train_epoch_start(self):
        if self.args.freeze_frontend:
            self.model.frontend.eval()  # keep BatchNorm running stats from LRS3
        if self.args.freeze_bn:
            # BatchNorm running stats update in train mode at any lr; on a few speakers (with
            # time-masked blank frames) they drift off LRS3's and hurt both domains (B2 GRID run:
            # lr 3e-6 for one epoch took GRID val WER 0.565 → 0.83)
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def training_step(self, batch, batch_idx):
        # stock step rescales by all_gather'd batch sizes for DDP (0-dim on one device); AdamW is
        # scale-invariant, so the plain mean loss trains the same
        return self._step(batch, batch_idx, "train")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        steps = len(self.trainer.datamodule.train_dataloader())
        sched = WarmupCosine(opt, self.args.warmup_epochs * steps, self.args.max_epochs * steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def greedy_texts(self, x, lengths) -> list[str]:
        mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        h = self.model.proj_encoder(self.model.frontend(x))
        h, _ = self.model.encoder(h, mask)
        best = self.model.ctc.ctc_lo(h).argmax(-1).cpu()
        return [self.text_transform.post_process(torch.tensor(collapse(best[i, :n].tolist()), dtype=torch.long))
                for i, n in enumerate(lengths.tolist())]

    def on_validation_epoch_start(self):
        self._errs = self._words = 0
        self.samples = []

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, step_type="val")
        hyps = self.greedy_texts(batch["inputs"], batch["input_lengths"])
        for tgt, hyp in zip(batch["targets"], hyps):
            ref = self.text_transform.post_process(tgt.cpu())
            self._errs += word_errors(ref, hyp)
            self._words += len(ref.split())
            self.samples.append((ref, hyp))
        return loss

    def on_validation_epoch_end(self):
        self.log("val_wer", self._errs / max(self._words, 1), prog_bar=True)


def build_args(a: argparse.Namespace, init_path: Path, val_file: str = "val.csv") -> Namespace:
    return Namespace(
        modality="video", root_dir=str(a.root), train_file="train.csv", val_file=val_file,
        test_file="test.csv", pretrained_model_path=str(init_path), transfer_frontend=False,
        transfer_encoder=False, lr=a.lr, weight_decay=a.weight_decay, warmup_epochs=a.warmup_epochs,
        max_epochs=a.epochs, max_frames=a.max_frames, ctc_weight=a.ctc_weight,
        freeze_frontend=a.freeze_frontend, freeze_bn=a.freeze_bn, exp_dir=str(a.exp_dir), exp_name=a.name,
    )


def nonempty(root: Path, csv: str) -> bool:
    p = root / "labels" / csv
    return p.is_file() and p.stat().st_size > 0


def evaluate(trainer: Trainer, module: FineTuneModule, a, init_path: Path, csv: str) -> dict | None:
    """Greedy WER of `module`'s current weights on ROOT/labels/<csv>."""
    if not nonempty(a.root, csv):
        return None
    dm = DataModule(build_args(a, init_path, val_file=csv), num_workers=a.num_workers)
    res = trainer.validate(module, datamodule=dm, verbose=False)[0]
    return {"wer": round(res["val_wer"], 4), "loss": round(res["loss_val"], 4),
            "samples": [{"ref": r, "hyp": h} for r, h in module.samples[:8]]}


def average(paths: list[str]) -> dict:
    avg = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[len("model."):]: v.float() if v.is_floating_point() else v for k, v in sd.items() if k.startswith("model.")}
        if avg is None:
            avg = sd
        else:
            for k in avg:
                avg[k] = avg[k] + sd[k] if avg[k].is_floating_point() else avg[k]
    return {k: v / len(paths) if v.is_floating_point() else v for k, v in avg.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="prepare_finetune_data.py output dir")
    ap.add_argument("--name", required=True, help="model name → checkpoints/<name>/, exp/<name>/")
    ap.add_argument("--init", type=Path, default=ML / "checkpoints" / STOCK / "model.pth",
                    help="Chaplin-layout weights to start from (default: stock 19.1)")
    ap.add_argument("--exp-dir", type=Path, default=ML / "exp")
    ap.add_argument("--ckpt-dir", type=Path, default=Path(os.environ.get("LIPREAD_CKPT_DIR", ML / "checkpoints")))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=1000, help="frames per batch (memory knob)")
    ap.add_argument("--freeze-frontend", action="store_true", help="train only encoder/decoder/CTC")
    ap.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm running stats from 19.1")
    ap.add_argument("--keep-ckpts", action="store_true", help="keep per-epoch .ckpt files (1 GB each)")
    ap.add_argument("--top-k", type=int, default=3, help="best-by-val-WER checkpoints to average")
    ap.add_argument("--precision", default="32-true", help="e.g. bf16-mixed on the 4090")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--limit-train-batches", type=float, default=1.0, help="debug: fraction/count")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not nonempty(a.root, "train.csv"):
        raise SystemExit(f"{a.root}/labels/train.csv missing or empty — run prepare_finetune_data.py")
    if not nonempty(a.root, "val.csv"):
        raise SystemExit("val.csv is empty: checkpoint selection needs val clips (use --val-frac)")
    seed_everything(a.seed, workers=True)
    out = a.exp_dir / a.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    init_path = out / "init_auto_avsr.pth"
    if not init_path.is_file():
        torch.save(convert_ckpt.rename(convert_ckpt.load_state_dict(a.init), to_auto_avsr=True), init_path)

    module = FineTuneModule(build_args(a, init_path))
    ckpt_cb = ModelCheckpoint(dirpath=out, monitor="val_wer", mode="min", save_top_k=a.top_k,
                              filename="{epoch}-{val_wer:.4f}", save_weights_only=True)
    trainer = Trainer(
        default_root_dir=out, max_epochs=a.epochs, accelerator=a.accelerator, devices=1,
        precision=a.precision, logger=CSVLogger(out, name="logs"), gradient_clip_val=10.0,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        limit_train_batches=a.limit_train_batches, log_every_n_steps=1, num_sanity_val_steps=0,
        enable_progress_bar=sys.stdout.isatty(),
    )
    summary: dict = {"name": a.name, "args": {k: str(v) for k, v in vars(a).items()},
                     "prep": json.loads((a.root / "prep_report.json").read_text()).get("counts")
                     if (a.root / "prep_report.json").is_file() else None}
    summary["stock"] = {s: evaluate(trainer, module, a, init_path, f"{s}.csv") for s in ("val", "test")}
    print(f"stock 19.1 greedy WER  val {summary['stock']['val']['wer']:.3f}"
          + (f"  test {summary['stock']['test']['wer']:.3f}" if summary["stock"]["test"] else ""))

    dm = DataModule(build_args(a, init_path), num_workers=a.num_workers)
    trainer.fit(module, datamodule=dm)
    trainer.strategy.optimizers = []  # free AdamW state (2x the weights) before loading candidates
    gc.collect()
    torch.cuda.empty_cache()
    summary["train_minutes"] = round((time.time() - t0) / 60, 1)

    best = sorted(ckpt_cb.best_k_models.items(), key=lambda kv: float(kv[1]))
    summary["best_k"] = [{"path": Path(p).name, "val_wer": round(float(w), 4)} for p, w in best]
    cand = {"best": torch.load(best[0][0], map_location="cpu", weights_only=False)["state_dict"]}
    cand["best"] = {k[len("model."):]: v for k, v in cand["best"].items() if k.startswith("model.")}
    if len(best) > 1:
        cand[f"avg{len(best)}"] = average([p for p, _ in best])
    scores = {}
    for tag, sd in cand.items():
        module.model.load_state_dict(sd)
        scores[tag] = evaluate(trainer, module, a, init_path, "val.csv")["wer"]
    pick = min(scores, key=scores.get)
    summary["candidates_val_wer"], summary["picked"] = scores, pick
    module.model.load_state_dict(cand[pick])
    summary["finetuned"] = {s: evaluate(trainer, module, a, init_path, f"{s}.csv") for s in ("val", "test")}
    torch.save(cand[pick], out / f"model_{pick}.pth")
    if not a.keep_ckpts:  # the picked weights are saved above; epoch ckpts fill a pod volume fast
        for p, _ in best:
            Path(p).unlink(missing_ok=True)
        init_path.unlink(missing_ok=True)

    # back to the served layout; verify both implementations give the same logits
    dst = a.ckpt_dir / a.name / "model.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    chaplin_sd = convert_ckpt.rename(cand[pick], to_auto_avsr=False)
    torch.save(chaplin_sd, dst)
    shutil.copy(ML / "checkpoints" / STOCK / "model.json", dst.parent / "model.json")
    while str(AUTO_AVSR) in sys.path:  # verify imports the Chaplin espnet first; it re-adds auto_avsr itself
        sys.path.remove(str(AUTO_AVSR))
    convert_ckpt.verify(chaplin_sd, cand[pick])
    summary["served_weights"] = str(dst)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    s, f = summary["stock"], summary["finetuned"]
    print(f"\npicked {pick} (val WER {scores})  → {dst}")
    for split in ("val", "test"):
        if s[split]:
            print(f"{split:5s} greedy WER  stock {s[split]['wer']:.3f}  →  fine-tuned {f[split]['wer']:.3f}")
    print(f"{summary['train_minutes']} min · LIPREAD_MODEL={a.name} uv run lipread transcribe clip.mp4")


if __name__ == "__main__":
    main()
