"""Fetch GRID audio-visual corpus clips (Cooke et al. 2006, CC BY 4.0, zenodo.org/records/3625687)
as `<speaker>_<nnn>.mpg` + `.txt` for prepare_finetune_data.py.

GRID = 34 speakers × 1000 frontal 3 s clips at 25 fps, fixed grammar
("BIN BLUE AT F TWO NOW"). Useful as a public rehearsal for B2 (new faces, webcam-like framing),
not as demo training data: its six-slot grammar would bias the model away from open speech.

    uv run python scripts/fetch_grid.py data/grid --speakers s1,s2,s4 --per-speaker 200
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ZENODO = "https://zenodo.org/records/3625687/files"


def download(name: str, cache: Path) -> Path:
    dst = cache / name
    if not dst.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"download {name}")
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(f"{ZENODO}/{name}?download=1") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(dst)
    return dst


def alignments(cache: Path) -> dict[tuple[str, str], str]:
    """(speaker, utterance id) → transcript, from alignments.zip (silences dropped)."""
    out = {}
    with zipfile.ZipFile(download("alignments.zip", cache)) as z:
        for n in z.namelist():
            if not n.endswith(".align") or "__MACOSX" in n or Path(n).name.startswith("._"):
                continue
            spk, utt = Path(n).parent.name, Path(n).stem
            words = [ln.split()[2] for ln in io.TextIOWrapper(z.open(n), errors="replace") if len(ln.split()) == 3]
            out[(spk, utt)] = " ".join(w for w in words if w not in ("sil", "sp")).upper()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--speakers", required=True, help="comma-separated, e.g. s1,s2,s4")
    ap.add_argument("--per-speaker", type=int, default=200)
    ap.add_argument("--cache", type=Path, default=None, help="where the zips go (default OUT/.zips)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = a.cache or a.out / ".zips"
    speakers = a.speakers.split(",")
    with ThreadPoolExecutor(8) as ex:  # zenodo is ~2 MB/s per stream; parallel streams add up
        list(ex.map(lambda n: download(n, cache), ["alignments.zip"] + [f"{s}.zip" for s in speakers]))
    text = alignments(cache)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for spk in speakers:
        zpath = download(f"{spk}.zip", cache)
        if not zipfile.is_zipfile(zpath):  # zenodo sometimes returns an error page; re-fetch next run
            print(f"{spk}: SKIP, {zpath.name} is not a zip (removed)")
            zpath.unlink()
            continue
        with zipfile.ZipFile(zpath) as z:
            all_vids = [n for n in z.namelist() if n.endswith(".mpg") and "__MACOSX" not in n]
            vids = sorted(n for n in all_vids if (spk, Path(n).stem) in text)
            # Zenodo's s11/s12/s15 alignments carry shifted utterance ids (0/1000 match their
            # videos); guessing the mapping would mislabel clips, so such speakers are skipped.
            if len(vids) < 0.9 * len(all_vids):
                print(f"{spk}: SKIP, only {len(vids)}/{len(all_vids)} videos have a matching alignment")
                continue
            pick = sorted(rng.sample(vids, min(a.per_speaker, len(vids))))
            for i, n in enumerate(pick):
                stem = f"{spk}_{i:04d}"
                with z.open(n) as src, open(a.out / f"{stem}.mpg", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                (a.out / f"{stem}.txt").write_text(text[(spk, Path(n).stem)] + "\n")
            print(f"{spk}: {len(pick)} of {len(vids)} clips")


if __name__ == "__main__":
    main()
