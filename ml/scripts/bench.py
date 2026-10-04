"""Benchmark the lip reader: WER (jiwer) + per-stage latency, against the local model or a
remote /lipread URL. Use it to compare local vs hosted tiers on the same clips.

Clip sources (pick one):
  --clips DIR            raw videos (*.mp4|*.webm|*.mov) with same-stem *.txt transcripts
  --lrs3-parquet FILE    HF mattymchen/lrs3-test shard — PRE-MADE 96x96 crops (no raw video),
                         sent as precropped; our face detection/crop is NOT exercised.
Backends:
  --backend local        in-process PyTorch LipReader (no HTTP)  [--device cuda:0|cpu]
  --backend onnx         exported encoder+CTC on onnxruntime, greedy only — the local/Electron tier
  --backend http --url   POST {url}/lipread (on-pod: http://127.0.0.1:8000, laptop: proxy URL)
    --transport mp4      multipart clip to /lipread (default; crops get mp4-encoded = lossy)
    --transport crops    raw uint8 crops to /lipread/crops, gzipped unless --no-gzip (lossless, what
                         the browser sends); raw videos are cropped here first, timed as client_crop

  uv run python scripts/bench.py --lrs3-parquet data/lrs3_test/0000.parquet --n 100 \
      --backend http --url http://127.0.0.1:8000 --decode greedy beam --out artifacts/bench

Writes <out>/<tag>.json (per-clip rows) and prints a markdown summary table.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import re
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import jiwer
import numpy as np

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


@dataclass
class Clip:
    id: str
    ref: str
    path: Path | None = None          # raw video
    crops: np.ndarray | None = None   # (T, 96, 96) uint8 pre-made crops


@dataclass
class Row:
    id: str
    decode: str
    ref: str
    hyp: str
    frames: int
    lat: dict = field(default_factory=dict)  # ms per stage
    error: str | None = None


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clips_from_dir(d: Path, n: int) -> list[Clip]:
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in VIDEO_EXT and p.with_suffix(".txt").is_file():
            out.append(Clip(p.stem, p.with_suffix(".txt").read_text().strip(), path=p))
    return out[:n]


def clips_from_parquet(f: Path, n: int) -> list[Clip]:
    import pyarrow.parquet as pq

    t = pq.ParquetFile(f).read_row_group(0, columns=["idx", "video", "label"]).slice(0, n)
    return [
        Clip(f"lrs3-{t.column('idx')[i].as_py()}", t.column("label")[i].as_py().strip(),
             crops=np.asarray(t.column("video")[i].as_py(), dtype=np.uint8))
        for i in range(len(t))
    ]


def crops_to_mp4(crops: np.ndarray, path: Path) -> None:
    from lipread.video import write_video
    write_video(path, crops)


class LocalBackend:
    def __init__(self, device: str | None, beam_size: int):
        from lipread.model import LipReader
        from lipread.preprocess import MouthCropper
        t = time.perf_counter()
        self.reader = LipReader(device=device, beam_size=beam_size)
        self.cropper = MouthCropper()
        self.load_s = time.perf_counter() - t
        self.name = f"local:{self.reader.device}"

    def run(self, clip: Clip, decode: str) -> Row:
        import torch

        from lipread.preprocess import precropped_patches, to_model_input
        from lipread.video import load_video_25fps
        t0 = time.perf_counter()
        frames = load_video_25fps(clip.path) if clip.path else None
        t1 = time.perf_counter()
        patches = self.cropper.crop(frames) if clip.path else precropped_patches(clip.crops)
        x = to_model_input(patches)
        t2 = time.perf_counter()
        res = self.reader.transcribe(x, decode=decode)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000  # noqa: E731
        return Row(clip.id, decode, clip.ref, res.text, int(x.shape[1]),
                   {"load": ms(t0, t1), "crop": ms(t1, t2), "vsr": ms(t2, t3), "total": ms(t0, t3)})


class OnnxBackend:
    """Tier 1 (local): exported encoder+CTC on onnxruntime + greedy CTC — what Electron would run."""

    def __init__(self, model_path: Path, providers: list[str] | None):
        import onnxruntime as ort

        from lipread.preprocess import MouthCropper
        t = time.perf_counter()
        avail = ort.get_available_providers()
        providers = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.tokens = json.loads((model_path.parent / "tokens.json").read_text())
        self.cropper = MouthCropper()
        self.load_s = time.perf_counter() - t
        self.name = f"onnx:{self.sess.get_providers()[0]}"

    def run(self, clip: Clip, decode: str) -> Row:
        from lipread.model import collapse_ctc, ids_to_text
        from lipread.preprocess import precropped_patches, to_model_input
        from lipread.video import load_video_25fps
        if decode != "greedy":
            raise SystemExit("onnx backend is greedy-only (beam search stays in Python)")
        t0 = time.perf_counter()
        frames = load_video_25fps(clip.path) if clip.path else None
        t1 = time.perf_counter()
        patches = self.cropper.crop(frames) if clip.path else precropped_patches(clip.crops)
        x = to_model_input(patches).unsqueeze(0).numpy()  # (1, 1, T, 88, 88)
        t2 = time.perf_counter()
        logp = self.sess.run(None, {"video": x})[0]
        text = ids_to_text(collapse_ctc(logp.argmax(-1).tolist()), self.tokens)
        t3 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000  # noqa: E731
        return Row(clip.id, decode, clip.ref, text, int(x.shape[2]),
                   {"load": ms(t0, t1), "crop": ms(t1, t2), "vsr": ms(t2, t3), "total": ms(t0, t3)})


class HttpBackend:
    def __init__(self, url: str, timeout: float, transport: str = "mp4", gzip_body: bool = True):
        import httpx
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        h = self.client.get(f"{self.url}/health").json()
        self.transport, self.gzip_body = transport, gzip_body
        self.cropper = None  # created on first raw video sent as crops
        via = f"/lipread/crops{' gzip' if gzip_body else ''}" if transport == "crops" else "/lipread mp4"
        self.name = f"http:{self.url} ({h.get('device')}) via {via}"
        self.health = h

    def run(self, clip: Clip, decode: str) -> Row:
        if self.transport == "crops":
            return self.run_crops(clip, decode)
        with tempfile.TemporaryDirectory() as d:
            if clip.path:
                path, precropped = clip.path, False
            else:
                path, precropped = Path(d) / f"{clip.id}.mp4", True
                crops_to_mp4(clip.crops, path)
            data = path.read_bytes()
        t0 = time.perf_counter()
        r = self.client.post(
            f"{self.url}/lipread",
            files={"file": (path.name, data, "video/mp4")},
            data={"decode": decode, "correct": "false", "precropped": str(precropped).lower()},
        )
        rtt = (time.perf_counter() - t0) * 1000
        return self.row(clip, decode, r, rtt, len(data))

    def run_crops(self, clip: Clip, decode: str) -> Row:
        """What the browser does: aligned uint8 crops, raw (optionally gzipped), to /lipread/crops."""
        from lipread.preprocess import MouthCropper, NoFaceError, precropped_patches
        from lipread.video import load_video_25fps
        client_lat = {}
        if clip.path:  # Python stand-in for the JS crop pipeline
            self.cropper = self.cropper or MouthCropper()
            t = time.perf_counter()
            try:
                crops = self.cropper.crop(load_video_25fps(clip.path))
            except NoFaceError:
                return Row(clip.id, decode, clip.ref, "", 0, error="no_face_detected (client-side crop)")
            client_lat["client_crop"] = (time.perf_counter() - t) * 1000
        else:
            crops = precropped_patches(clip.crops)
        n, h, w = crops.shape
        data = np.ascontiguousarray(crops, dtype=np.uint8).tobytes()
        headers = {"Content-Type": "application/octet-stream"}
        if self.gzip_body:
            data, headers["Content-Encoding"] = gzip.compress(data, compresslevel=6), "gzip"
        t0 = time.perf_counter()
        r = self.client.post(f"{self.url}/lipread/crops", content=data, headers=headers,
                             params={"t": n, "h": h, "w": w, "decode": decode, "correct": "false"})
        rtt = (time.perf_counter() - t0) * 1000
        return self.row(clip, decode, r, rtt, len(data), client_lat)

    @staticmethod
    def row(clip: Clip, decode: str, r, rtt: float, upload_bytes: int, client_lat: dict | None = None) -> Row:
        if r.status_code != 200:
            return Row(clip.id, decode, clip.ref, "", 0, {"rtt": rtt}, error=r.text[:200])
        j = r.json()
        lat = {k: float(v) for k, v in j["latency_ms"].items()}
        lat["rtt"] = rtt
        lat["network"] = rtt - lat["total"]
        lat["upload_kb"] = upload_bytes / 1024
        lat.update(client_lat or {})
        return Row(clip.id, decode, clip.ref, j["raw_text"], j.get("frames", 0), lat)


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))] if xs else float("nan")


def summarize(rows: list[Row]) -> dict:
    ok = [r for r in rows if r.error is None]
    out = {"n": len(rows), "errors": len(rows) - len(ok)}
    if ok:
        out["wer"] = jiwer.wer([norm(r.ref) for r in ok], [norm(r.hyp) or "<empty>" for r in ok])
        out["mean_seconds"] = statistics.mean(r.frames for r in ok) / 25
        for k in ok[0].lat:
            vals = [r.lat[k] for r in ok if k in r.lat]
            out[f"{k}_p50"], out[f"{k}_p95"] = pct(vals, 0.5), pct(vals, 0.95)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips", type=Path)
    src.add_argument("--lrs3-parquet", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--backend", choices=["local", "onnx", "http"], default="local")
    ap.add_argument("--url")
    ap.add_argument("--transport", choices=["mp4", "crops"], default="mp4",
                    help="http backend: mp4 multipart to /lipread, or raw crops to /lipread/crops (lossless)")
    ap.add_argument("--gzip", action=argparse.BooleanOptionalAction, default=True,
                    help="gzip /lipread/crops bodies, as the browser does (default: on)")
    ap.add_argument("--onnx-path", type=Path, default=Path("artifacts/lipread_ctc.onnx"))
    ap.add_argument("--providers", nargs="+", help="onnxruntime EPs (default: CUDA if available, else CPU)")
    ap.add_argument("--device")
    ap.add_argument("--decode", nargs="+", choices=["greedy", "beam"], default=["greedy"])
    ap.add_argument("--beam-size", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--warmup", type=int, default=2, help="untimed requests before measuring")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/bench"))
    a = ap.parse_args()

    clips = clips_from_dir(a.clips, a.n) if a.clips else clips_from_parquet(a.lrs3_parquet, a.n)
    if not clips:
        raise SystemExit("no clips found")
    if a.backend == "http":
        if not a.url:
            raise SystemExit("--backend http needs --url")
        be = HttpBackend(a.url, a.timeout, a.transport, a.gzip)
    elif a.backend == "onnx":
        be = OnnxBackend(a.onnx_path, a.providers)
    else:
        be = LocalBackend(a.device, a.beam_size)
    source = "raw video" if a.clips else "pre-made crops (precropped)"
    print(f"# {len(clips)} clips from {a.clips or a.lrs3_parquet} [{source}] → {be.name}")

    tag = a.tag or ("http-crops" if a.backend == "http" and a.transport == "crops" else a.backend)
    summaries = {}
    for decode in a.decode:
        for c in clips[: a.warmup]:
            be.run(c, decode)
        rows = []
        for i, c in enumerate(clips, 1):
            rows.append(be.run(c, decode))
            if i % 25 == 0:
                print(f"  {decode}: {i}/{len(clips)}", flush=True)
        summaries[decode] = summarize(rows)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{tag}-{decode}.json").write_text(json.dumps({
            "backend": be.name, "source": source, "decode": decode, "host": platform.node(),
            "summary": summaries[decode], "rows": [r.__dict__ for r in rows],
        }, indent=1))

    stages = [k[:-4] for k in next(iter(summaries.values())) if k.endswith("_p50")]
    unit = lambda s: "KB" if s.endswith("_kb") else "ms"  # noqa: E731
    print(f"\n| decode | n | err | WER | " + " | ".join(f"{s} p50/p95 {unit(s)}" for s in stages) + " |")
    print("|---" * (4 + len(stages)) + "|")
    for d, s in summaries.items():
        cells = " | ".join(f"{s[f'{k}_p50']:.0f} / {s[f'{k}_p95']:.0f}" for k in stages)
        print(f"| {d} | {s['n']} | {s['errors']} | {s.get('wer', float('nan')):.1%} | {cells} |")


if __name__ == "__main__":
    main()
