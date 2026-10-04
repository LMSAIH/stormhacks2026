"""Team clips → auto_avsr custom-dataset (`cstm`) layout for fine-tuning (B2).

Input dir: `<speaker>_<nnn>.{mp4,webm,mov,mkv,avi}` + `<speaker>_<nnn>.txt` (the exact words), or
`<speaker>_<nnn>.npy` (T, 96, 96) uint8 crops that are already aligned (LRS3 mirror, Phase 1 only).
Videos go through *our* `MouthCropper` at 25 fps (D36), i.e. exactly what the browser and server
feed the model; clips failing the face-coverage gate are skipped and logged.

    uv run python scripts/prepare_finetune_data.py prepare data/recordings data/ft \
        --holdout-speaker alice [--val-frac 0.1] [--preview]
    # Phase 1 only: dump LRS3-test crops as fake speakers (never claim test-set gains from these)
    uv run python scripts/prepare_finetune_data.py dump-lrs3 data/lrs3_test/0000.parquet \
        data/lrs3_tiny --start 600 --n 12 --speakers 3

Output under ROOT:
    cstm/cstm_video/<stem>.npy   (T, 96, 96) uint8 gray crops (lossless; the mp4 round trip
                                 auto_avsr uses would add codec noise the live crops don't have)
    cstm/cstm_video/<stem>.mp4   preview of the same crops (--preview), for eyeballing
    cstm/cstm_text/<stem>.txt    normalised transcript
    labels/{train,val,test}.csv  `cstm,cstm_video/<stem>.npy,<frames>,<token ids>`
    prep_report.json             per-clip status, skips with reasons, split, counts
Split is by speaker: the whole --holdout-speaker goes to `test` (unseen face), ~--val-frac of each
other speaker's clips to `val`.

--pairs DIR adds the app's opt-in training pairs (`POST /training-pairs`: `<id>.npz` crops +
`<id>.txt` + `<id>.json`) to `train` only, never val/test, filtered by --pair-sources. Those pairs
may live in a public dataset (D61); keep team recordings in their own private one.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
SPM_DIR = ML / "third_party" / "auto_avsr" / "spm" / "unigram"
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".mpg"}
STEM_RE = re.compile(r"^(?P<speaker>[A-Za-z0-9-]+)_(?P<num>\d+)$")
MAX_FRAMES = 600  # 24 s at 25 fps; train.py batches by frame count (--max-frames 1600 default)


class Tokenizer:
    """Same mapping as auto_avsr `datamodule.transforms.TextTransform` (unigram5000, blank=0)."""

    def __init__(self) -> None:
        import sentencepiece

        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(SPM_DIR / "unigram5000.model"))
        units = (SPM_DIR / "unigram5000_units.txt").read_text(encoding="utf8").splitlines()
        self.hashmap = {u.split()[0]: u.split()[-1] for u in units}

    def ids(self, text: str) -> list[int]:
        return [int(self.hashmap.get(p, self.hashmap["<unk>"])) for p in self.spm.EncodeAsPieces(text)]


def normalise(text: str) -> str:
    """LRS3 label style: uppercase words, apostrophes kept, everything else dropped."""
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_clips(src: Path) -> list[tuple[str, str, Path, Path]]:
    """(stem, speaker, media, txt) for every media file with a matching .txt."""
    out = []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in VIDEO_EXT | {".npy"}:
            continue
        m = STEM_RE.match(p.stem)
        txt = p.with_suffix(".txt")
        if not m or not txt.is_file():
            print(f"ignore {p.name}: {'no matching .txt' if m else 'name is not <speaker>_<nnn>'}")
            continue
        out.append((p.stem, m["speaker"].lower(), p, txt))
    return out


def split_clips(stems_by_speaker: dict[str, list[str]], holdout: set[str], val_frac: float,
                seed: int, train_only: frozenset[str] = frozenset()) -> dict[str, str]:
    rng = random.Random(seed)
    split = {}
    for spk, stems in sorted(stems_by_speaker.items()):
        if spk in train_only:
            split.update({s: "train" for s in stems})
            continue
        if spk in holdout:
            split.update({s: "test" for s in stems})
            continue
        stems = sorted(stems)
        # at least one val clip per speaker (val WER picks the checkpoint), but keep most for train
        n_val = max(1, round(len(stems) * val_frac)) if val_frac > 0 and len(stems) >= 3 else 0
        val = set(rng.sample(stems, n_val))
        split.update({s: "val" if s in val else "train" for s in stems})
    return split


PAIRS_SPEAKER = "pairs"


def pair_patches(crops: np.ndarray) -> np.ndarray:
    """Training-pair crops (T, 96|88, 96|88) uint8 → (T, 96, 96).

    The browser may send the 88×88 centre crop. Resizing it to 96 would zoom the mouth by 9% against
    inference, so pad 4 px by edge reflection instead: the centre 88 (what inference sees) is exact,
    and train-time RandomCrop(88) only ever shows a few reflected pixels at the border.
    """
    if crops.ndim != 3 or crops.shape[1] != crops.shape[2] or crops.shape[1] not in (88, 96):
        raise ValueError(f"pair crops must be (T, 96|88, 96|88), got {crops.shape}")
    if crops.shape[1] == 88:
        crops = np.pad(crops, ((0, 0), (4, 4), (4, 4)), mode="reflect")
    return np.ascontiguousarray(crops, dtype=np.uint8)


def find_pairs(src: Path, sources: set[str]) -> list[tuple[str, str, Path, Path]]:
    """(stem, speaker, npz, txt) for each training pair whose .json `source` is in `sources`."""
    out = []
    for npz in sorted(src.glob("*.npz")):
        txt, meta = npz.with_suffix(".txt"), npz.with_suffix(".json")
        if not txt.is_file():
            continue
        source = json.loads(meta.read_text()).get("source", "?") if meta.is_file() else "?"
        if source in sources:
            out.append((f"{PAIRS_SPEAKER}_{npz.stem}", PAIRS_SPEAKER, npz, txt))
    return out


_CROPPER = None


def _init_worker(min_face_coverage: float) -> None:
    global _CROPPER
    from lipread.preprocess import MouthCropper

    _CROPPER = MouthCropper(min_face_coverage=min_face_coverage)


def _crop_one(job) -> tuple[np.ndarray | None, str | None]:
    """(stem, speaker, media, text) → ((T, 96, 96) uint8 crops, None) or (None, reason)."""
    from lipread.preprocess import NoFaceError, precropped_patches
    from lipread.video import load_video_25fps

    media = job[2]
    try:
        if media.suffix == ".npz":
            return pair_patches(np.load(media)["crops"]), None
        if media.suffix == ".npy":
            return precropped_patches(np.load(media)), None
        return _CROPPER.crop(load_video_25fps(media)), None
    except (NoFaceError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def prepare(a: argparse.Namespace) -> None:
    from lipread.video import write_video

    clips = find_clips(a.src)
    if not clips:
        raise SystemExit(f"no <speaker>_<nnn>.(mp4|…|npy) + .txt pairs in {a.src}")
    speakers = sorted({c[1] for c in clips})
    holdout = {h.strip().lower() for h in a.holdout_speaker.split(",")} if a.holdout_speaker else set()
    if holdout - set(speakers):
        raise SystemExit(f"--holdout-speaker {sorted(holdout - set(speakers))} not among speakers {speakers}")
    if not holdout and len(speakers) > 1:
        print("warning: no --holdout-speaker; test.csv will be empty (no unseen-face check)")
    if PAIRS_SPEAKER in speakers:
        raise SystemExit(f"speaker id {PAIRS_SPEAKER!r} is reserved for --pairs")
    if a.pairs:
        pairs = find_pairs(a.pairs, set(a.pair_sources.split(",")))
        print(f"{len(pairs)} training pairs from {a.pairs} (sources {a.pair_sources}) → train only")
        clips += pairs

    vid_dir, txt_dir, lab_dir = a.root / "cstm" / "cstm_video", a.root / "cstm" / "cstm_text", a.root / "labels"
    for d in (vid_dir, txt_dir, lab_dir):
        d.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer()
    rows, report = {}, {"clips": {}, "skipped": {}}
    jobs = []
    for stem, spk, media, txt in clips:
        text = normalise(txt.read_text(encoding="utf8"))
        if text:
            jobs.append((stem, spk, media, text))
        else:
            report["skipped"][stem] = "empty transcript"
    with Pool(a.workers, initializer=_init_worker, initargs=(a.min_face_coverage,)) if a.workers > 1 \
            else nullcontext() as pool:
        if pool is None:
            _init_worker(a.min_face_coverage)
        results = pool.imap(_crop_one, jobs, chunksize=4) if pool else map(_crop_one, jobs)
        for (stem, spk, _, text), (patches, err) in zip(jobs, results):
            if err:
                report["skipped"][stem] = err
                print(f"skip {stem}: {err}")
                continue
            ids = tok.ids(text)
            t = len(patches)
            # The conformer subsamples nothing in time, but CTC still needs T >= len(targets).
            if t > MAX_FRAMES or t < len(ids) or t < 10:
                report["skipped"][stem] = f"bad length: {t} frames for {len(ids)} tokens"
                print(f"skip {stem}: {t} frames for {len(ids)} tokens")
                continue
            np.save(vid_dir / f"{stem}.npy", np.ascontiguousarray(patches, dtype=np.uint8))
            if a.preview:
                write_video(vid_dir / f"{stem}.mp4", patches)
            (txt_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf8")
            rows[stem] = (spk, f"cstm,cstm_video/{stem}.npy,{t},{' '.join(map(str, ids))}")
            report["clips"][stem] = {"speaker": spk, "frames": t, "tokens": len(ids), "text": text}
            print(f"ok   {stem}: {t} frames, {len(ids)} tokens  {text}")

    by_spk = defaultdict(list)
    for stem, (spk, _) in rows.items():
        by_spk[spk].append(stem)
    split = split_clips(by_spk, holdout, a.val_frac, a.seed, train_only=frozenset({PAIRS_SPEAKER}))
    for name in ("train", "val", "test"):
        lines = [rows[s][1] for s in sorted(rows) if split[s] == name]
        (lab_dir / f"{name}.csv").write_text("".join(f"{line}\n" for line in lines))
    for s in rows:
        report["clips"][s]["split"] = split[s]

    counts = {n: sum(1 for s in split.values() if s == n) for n in ("train", "val", "test")}
    frames = {n: sum(report["clips"][s]["frames"] for s in rows if split[s] == n) for n in counts}
    report.update(src=str(a.src), holdout_speaker=sorted(holdout), speakers=speakers, counts=counts,
                  seconds={n: round(f / 25, 1) for n, f in frames.items()},
                  n_skipped=len(report["skipped"]))
    (a.root / "prep_report.json").write_text(json.dumps(report, indent=1))
    print(f"\n{len(rows)} clips kept, {len(report['skipped'])} skipped; split {counts}; "
          f"seconds {report['seconds']} → {a.root}")
    if not counts["train"]:
        raise SystemExit("no training clips")


def dump_lrs3(a: argparse.Namespace) -> None:
    """Write LRS3-test mirror crops as `<fake speaker>_<nnn>.npy` + `.txt` (pipeline tests only)."""
    import pyarrow.parquet as pq

    t = pq.ParquetFile(a.parquet).read_row_group(0, columns=["idx", "video", "label"])
    t = t.slice(a.start, a.n)
    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(t.num_rows):
        spk = f"lrs{i % a.speakers}"
        stem = f"{spk}_{t.column('idx')[i].as_py():04d}"
        np.save(a.out / f"{stem}.npy", np.asarray(t.column("video")[i].as_py(), dtype=np.uint8))
        (a.out / f"{stem}.txt").write_text(t.column("label")[i].as_py().strip() + "\n")
    print(f"wrote {t.num_rows} clips ({a.speakers} fake speakers) → {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="clips dir → cstm layout + label CSVs")
    p.add_argument("src", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("--holdout-speaker", help="whole speaker(s), comma-separated → test.csv (unseen faces)")
    p.add_argument("--workers", type=int, default=1, help="parallel cropping processes")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--min-face-coverage", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true", help="also write an mp4 of each crop")
    p.add_argument("--pairs", type=Path, help="dir of app training pairs (<id>.npz/.txt/.json) → train only")
    p.add_argument("--pair-sources", default="typed,picked",
                   help="which pair sources to use: typed (user typed the fix), picked (top-3 pick), "
                        "accepted (left as read; mostly the model's own output)")
    p.set_defaults(fn=prepare)
    d = sub.add_parser("dump-lrs3", help="LRS3-test parquet crops → <speaker>_<nnn>.npy + .txt")
    d.add_argument("parquet", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--start", type=int, default=600, help="skip LRS3-100 (the regression gate set)")
    d.add_argument("--n", type=int, default=12)
    d.add_argument("--speakers", type=int, default=3)
    d.set_defaults(fn=dump_lrs3)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
