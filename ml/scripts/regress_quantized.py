"""Regression gate for the quantized lip-reading ONNX model.

Runs the fp32 reference (artifacts/lipread_ctc.onnx) and a quantized variant on IDENTICAL,
once-preprocessed inputs with onnxruntime's CPU EP, greedy-decodes both, and enforces:

  gates  size <= 220 MB · WER_q <= WER_fp32 + 1.0 pt on each set · mean per-clip argmax-frame
         agreement vs fp32 >= 0.95 · no clip where quantized is empty but fp32 is not
         (+ invariants: I/O contract unchanged, every clip preprocessed, run and finite; and, on full
         runs, a 250-frame input (the app's 10 s cap) runs: guards trimmed position tables and shapes)
  lock   the model's sha256 and its exact greedy texts for a FIXED subset (first 10 LRS3 rows +
         first 5 raw clips by name) must equal tests/quantized_baseline.json. ORT CPU is
         deterministic, so any diff means the model, onnxruntime, the CPU class, or preprocessing
         changed. Intended change: review, then --update-baseline and commit the JSON.

Sets: lrs3 = data/lrs3_test/0000.parquet (pre-made 96x96 crops: exercises model + decoding only);
      raw  = data/raw_eval/*.mp4 + .txt (raw face video: also exercises load -> MediaPipe crop).

  uv run python scripts/regress_quantized.py                    # full: 100 LRS3 + 20 raw, ~2-4 min
  uv run python scripts/regress_quantized.py --fast             # 5 LRS3 + 2 raw (what smoke.sh runs)
  uv run python scripts/regress_quantized.py --update-baseline  # lock this model after a passing run

Prints a markdown summary (stdout; progress goes to stderr) and writes a JSON report with per-clip
rows. Exit code: 0 = every gate and the lock pass, 1 = a gate or the lock failed, 2 = bad setup.
Importable: scripts/smoke_checks.py calls evaluate() in-process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `import bench` must work however this module is imported
    sys.path.insert(0, str(HERE))

import bench  # noqa: E402  Clip, clips_from_dir, norm, pct are reused rather than duplicated

ML = HERE.parent
ART = ML / "artifacts"
FP32 = ART / "lipread_ctc.onnx"
BASELINE = ML / "tests" / "quantized_baseline.json"
LRS3_PARQUET = ML / "data" / "lrs3_test" / "0000.parquet"
RAW_DIR = ML / "data" / "raw_eval"

# Gate thresholds: the contract this script enforces.
MAX_SIZE_MB = 220.0        # decimal MB (1e6 bytes), like export_onnx.py
MAX_WER_DELTA_PTS = 1.0    # absolute WER points per set; never tighter than one word error (see gates)
MIN_MEAN_AGREEMENT = 0.95  # mean over clips of the fraction of frames whose argmax matches fp32

SPLITS = ("lrs3", "raw")
FULL = {"lrs3": 100, "raw": 20}
FAST = {"lrs3": 5, "raw": 2}
FIXED = {"lrs3": 10, "raw": 5}  # clips whose quantized texts are locked in the baseline
IN_NAME, OUT_NAME = "video", "log_probs"
LONG_FRAMES = 250  # the app caps an utterance at 10 s = 250 frames @ 25 fps; full runs probe that length
SCHEMA = 1
ENV_KEYS = ("onnxruntime", "cpu", "isa", "threads", "numpy", "torch", "cv2", "mediapipe")
MAX_SHOWN = 12  # diffs printed before "... and N more"


class SetupError(Exception):
    """A required input is missing or the options are inconsistent: exit code 2, not a regression."""


class ModelLoadError(RuntimeError):
    """onnxruntime rejects a model: exit code 1 (a quantization that produces an unloadable graph is a regression)."""


@dataclass
class Options:
    model: Path
    reference: Path = FP32
    tokens: Path | None = None         # default: tokens.json next to the reference
    lrs3: int = FULL["lrs3"]
    raw: int = FULL["raw"]
    baseline: Path | None = BASELINE   # None: no lock check
    fast: bool = False                 # label only (report name); the clip counts are lrs3/raw
    update_baseline: bool = False      # lock rows become informational; the --fast subset is vetted too
    threads: int = 0                   # onnxruntime intra-op threads; 0 = ORT's default
    quiet: bool = False                # no progress on stderr (in-process callers)


@dataclass
class ClipResult:
    id: str
    split: str
    ref: str
    frames: int = 0
    text_fp32: str = ""
    text_q: str = ""
    agree: float | None = None
    ms_fp32: float | None = None
    ms_q: float | None = None
    input_sha256: str = ""  # fingerprint of the model input: tells preprocessing drift from model drift
    error: str | None = None


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ML.parent))
    except ValueError:
        return str(p)


# ───────────────────────────── model files, environment ─────────────────────────────

def find_quantized(art: Path = ART) -> Path | None:
    """Newest artifacts/lipread_ctc.<variant>.onnx; the fp32 export lipread_ctc.onnx never matches."""
    cands = [p for p in art.glob("lipread_ctc.*.onnx") if p.is_file() and p.name != FP32.name]
    return max(cands, key=lambda p: p.stat().st_mtime, default=None)


def locked_or_newest(baseline: Path | None = BASELINE, art: Path = ART) -> Path | None:
    """The model the baseline locks if it is in artifacts/, else the newest variant.

    smoke.sh uses this so experimental variants lying around don't fail a lock made for another file.
    """
    if baseline and baseline.is_file():
        try:
            locked = art / json.loads(baseline.read_text(encoding="utf-8"))["model"]["file"]
        except (OSError, ValueError, KeyError, TypeError):
            locked = None
        if locked and locked.is_file():
            return locked
    return find_quantized(art)


def file_info(path: Path, sha: bool = True) -> dict:
    """File name, size and sha256. An external-data sidecar `<model>.data` counts towards both
    (the sha of a model with a sidecar is the sha256 of the two hex digests concatenated)."""
    files = [path] + [s for s in (Path(f"{path}.data"),) if s.is_file()]
    size = sum(f.stat().st_size for f in files)
    digest = None
    if sha:
        parts = []
        for f in files:
            with open(f, "rb") as fh:
                parts.append(hashlib.file_digest(fh, "sha256").hexdigest())
        digest = parts[0] if len(parts) == 1 else hashlib.sha256("".join(parts).encode()).hexdigest()
    return {"file": path.name, "sha256": digest, "size_bytes": size, "size_mb": round(size / 1e6, 2)}


_ISA = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "fma")


def cpu_info() -> tuple[str, str]:
    """(CPU model, ISA features that pick int8 kernels). Locked texts are per CPU class: VNNI and
    vpmaddubsw u8s8 paths round differently, so a CPU change can legitimately flip a marginal frame."""
    name, flags = platform.processor() or platform.machine(), ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            key, _, val = (s.strip() for s in line.partition(":"))
            if key == "model name":
                name = val
            elif key == "flags":
                flags = " ".join(f for f in _ISA if f in val.split())
                break
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
            name = out or name
        except (OSError, subprocess.SubprocessError):
            pass
    return name, flags


def env_info(threads: int) -> dict:
    import onnxruntime as ort

    cpu, isa = cpu_info()
    libs = {m: getattr(sys.modules.get(m), "__version__", None) for m in ("numpy", "torch", "cv2", "mediapipe")}
    return {"onnxruntime": ort.__version__, "cpu": cpu, "isa": isa, "threads": threads or "default",
            **{k: v for k, v in libs.items() if v}, "os": platform.platform(), "python": platform.python_version()}


def open_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # errors only: ORT warnings would bury the report
    if threads:
        so.intra_op_num_threads = threads
    try:
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001  invalid graph, unsupported op/dtype for the CPU EP ...
        raise ModelLoadError(f"onnxruntime cannot load {rel(path)}: {' '.join(str(e).split())[:300]}") from e


def io_problems(sess, vocab: int) -> list[str]:
    """Deviations from the I/O the frontend codes against: video f32 [1,1,T,88,88] -> log_probs f32 [T,vocab]."""
    bad = []
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if (len(ins) != 1 or ins[0].name != IN_NAME or ins[0].type != "tensor(float)"
            or len(ins[0].shape) != 5 or list(ins[0].shape[3:]) != [88, 88]):
        bad.append(f"inputs {[(i.name, i.type, i.shape) for i in ins]}")
    if (len(outs) != 1 or outs[0].name != OUT_NAME or outs[0].type != "tensor(float)"
            or len(outs[0].shape) != 2 or outs[0].shape[1] != vocab):
        bad.append(f"outputs {[(o.name, o.type, o.shape) for o in outs]}")
    return bad


# ───────────────────────────── data ─────────────────────────────

def load_lrs3(n: int) -> list[bench.Clip]:
    """First n LRS3 rows as pre-made crops, built exactly like bench.clips_from_parquet. That helper
    decodes the whole 346 MB row group (6.5 s even for 5 clips); iter_batches stops after one batch."""
    import pyarrow.parquet as pq

    if n <= 0:
        return []
    batch = next(pq.ParquetFile(LRS3_PARQUET).iter_batches(batch_size=n, columns=["idx", "video", "label"]))
    return [
        bench.Clip(f"lrs3-{batch.column('idx')[i].as_py()}", batch.column("label")[i].as_py().strip(),
                   crops=np.asarray(batch.column("video")[i].as_py(), dtype=np.uint8))
        for i in range(len(batch))
    ]


def missing_inputs(o: Options) -> str:
    """'' when everything needed exists, else a one-line description of what is missing."""
    miss = []
    tokens = o.tokens or o.reference.parent / "tokens.json"
    for what, p in (("quantized model", o.model), ("fp32 reference", o.reference), ("tokens", tokens)):
        if not Path(p).is_file():
            miss.append(f"{what} {rel(p)}")
    if o.lrs3 > 0 and not LRS3_PARQUET.is_file():
        miss.append(f"LRS3 shard {rel(LRS3_PARQUET)} (--lrs3 0 skips the set)")
    if o.raw > 0 and not (RAW_DIR.is_dir() and bench.clips_from_dir(RAW_DIR, 1)):
        miss.append(f"raw eval clips in {rel(RAW_DIR)} (--raw 0 skips the set)")
    return "missing: " + ", ".join(miss) if miss else ""


# ───────────────────────────── metrics and gates ─────────────────────────────

def summarize_set(rows: list[ClipResult]) -> dict:
    import jiwer

    ok = [r for r in rows if r.error is None]
    s = {"n": len(rows), "n_ok": len(ok), "errors": len(rows) - len(ok),
         "error_examples": [f"{r.id}: {r.error}" for r in rows if r.error][:3]}
    if not ok:
        return s
    hyp = lambda t: bench.norm(t) or "<empty>"  # noqa: E731  same empty-hypothesis rule as bench.summarize
    pairs = [(bench.norm(r.ref), r) for r in ok]
    pairs = [(ref, r) for ref, r in pairs if ref]  # jiwer rejects empty references
    refs = [ref for ref, _ in pairs]
    s["ref_words"] = sum(len(ref.split()) for ref in refs)
    s["wer_fp32_pct"] = 100 * jiwer.wer(refs, [hyp(r.text_fp32) for _, r in pairs])
    s["wer_q_pct"] = 100 * jiwer.wer(refs, [hyp(r.text_q) for _, r in pairs])
    s["wer_delta_pts"] = s["wer_q_pct"] - s["wer_fp32_pct"]
    s["wer_q_vs_fp32_pct"] = 100 * jiwer.wer([hyp(r.text_fp32) for r in ok], [hyp(r.text_q) for r in ok])
    agree = [r.agree for r in ok]
    s["agree_mean"], s["agree_min"] = float(np.mean(agree)), float(np.min(agree))
    s["agree_min_clip"] = ok[int(np.argmin(agree))].id
    s["text_diff"] = sum(r.text_q != r.text_fp32 for r in ok)
    s["empty_regress"] = sum(not r.text_q.strip() and bool(r.text_fp32.strip()) for r in ok)
    s["mean_frames"] = float(np.mean([r.frames for r in ok]))
    for tag, key in (("fp32", "ms_fp32"), ("q", "ms_q")):
        v = [getattr(r, key) for r in ok]
        s[f"lat_{tag}_p50"], s[f"lat_{tag}_p95"] = bench.pct(v, 0.5), bench.pct(v, 0.95)
    return s


def build_gates(model: dict, contract: dict, sets: dict, lock: dict | None, enforce_lock: bool = True,
                long_probe: dict | None = None) -> list[dict]:
    """Gate rows: {name, scope, value, limit, status PASS|FAIL|SKIP|INFO}."""
    gates: list[dict] = []

    def add(name: str, scope: str, value: str, limit: str, ok: bool, status: str | None = None) -> None:
        gates.append({"name": name, "scope": scope, "value": value, "limit": limit,
                      "status": status or ("PASS" if ok else "FAIL")})

    problems = [f"{tag}: {p}" for tag, ps in contract.items() for p in ps]
    add("io contract", "model", "unchanged" if not problems else "; ".join(problems),
        f"{IN_NAME} f32 [1,1,T,88,88] -> {OUT_NAME} f32 [T,vocab]", not problems)
    size_mb = model["size_bytes"] / 1e6
    add("size_mb", "model", f"{size_mb:.1f}", f"<= {MAX_SIZE_MB:g}", size_mb <= MAX_SIZE_MB)
    if long_probe:
        add(f"max-length input ({long_probe['frames']} frames)", "model",
            long_probe["problem"] or f"runs, shape ok, finite (argmax agree {long_probe['agree']:.3f})",
            "runs, [T,vocab], finite", long_probe["problem"] is None)
    for split, s in sets.items():
        add("clips run ok", split, f"{s['n_ok']}/{s['n']}" + (f" ({s['error_examples'][0][:90]})" if s["errors"] else ""),
            "all", s["errors"] == 0)
        if not s["n_ok"]:
            continue
        # 1.0 pt is finer than one word error on a handful of clips (--fast: 1 word = ~3-8 pts), so the
        # tolerance is never tighter than one word. On >= 100 reference words (the full sets) it is 1.0.
        word_pts = 100.0 / s["ref_words"] if s["ref_words"] else 100.0
        tol = max(MAX_WER_DELTA_PTS, word_pts)
        limit = f"<= +{tol:.2f} pts" + (f" (1 word = {word_pts:.1f} pts)" if tol > MAX_WER_DELTA_PTS else "")
        add("WER delta vs fp32", split, f"{s['wer_delta_pts']:+.2f} pts", limit, s["wer_delta_pts"] <= tol + 1e-9)
        add("mean argmax agreement", split, f"{s['agree_mean']:.4f}", f">= {MIN_MEAN_AGREEMENT}",
            s["agree_mean"] >= MIN_MEAN_AGREEMENT)
        add("quant empty, fp32 not", split, str(s["empty_regress"]), "== 0", s["empty_regress"] == 0)
    if lock is None:
        add("lock: baseline", "lock", "no baseline file", "-", True, "SKIP")
    else:
        info = None if enforce_lock else "INFO"
        add("lock: model sha256", "lock",
            "match" if lock["sha_ok"] else f"changed {lock['locked_sha256'][:8]} -> {lock['model_sha256'][:8]}",
            "== baseline", lock["sha_ok"], info)
        n_same = lock["compared"] - len(lock["diffs"])
        add("lock: fixed-subset texts", "lock",
            f"{n_same}/{lock['compared']} identical" + (f", {len(lock['missing'])} missing" if lock["missing"] else "")
            + (f" (of {lock['locked_total']} locked)" if lock["compared"] < lock["locked_total"] else ""),
            "all identical", not lock["diffs"] and not lock["missing"] and lock["compared"] > 0, info)
    return gates


# ───────────────────────────── baseline lock ─────────────────────────────

def load_baseline(path: Path) -> dict:
    try:
        b = json.loads(Path(path).read_text(encoding="utf-8"))
        if b["schema"] != SCHEMA or not {"model", "clips", "env"} <= b.keys():
            raise ValueError(f"schema {b['schema']!r}, expected {SCHEMA}")
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SetupError(f"cannot use baseline {rel(path)} ({type(e).__name__}: {e}); "
                         "re-lock with --update-baseline") from e
    return b


def compare_lock(base: dict, model: dict, reference: dict, rows: list[ClipResult], env: dict,
                 ran: dict[str, int]) -> dict:
    """Compare this run with the baseline. Only locked clips this run covers are compared (--fast covers
    a prefix of the locked subset); a covered clip that can't be found counts as a failure."""
    by_id = {r.id: r for r in rows}
    diffs, missing, inputs_changed, compared = [], [], [], 0
    for split in SPLITS:
        for cid, rec in list(base["clips"].get(split, {}).items())[: ran.get(split, 0)]:
            r = by_id.get(cid)
            if r is None:
                missing.append(cid)
                continue
            compared += 1
            now = r.text_q if r.error is None else f"<{r.error}>"
            if now != rec["text"]:
                diffs.append({"id": cid, "split": split, "locked": rec["text"], "now": now,
                              "fp32": r.text_fp32, "ref": r.ref})
            if r.input_sha256 and rec.get("input_sha256") and r.input_sha256 != rec["input_sha256"]:
                inputs_changed.append(cid)
    base_env = base.get("env", {})
    env_changed = [(k, base_env.get(k), env.get(k)) for k in ENV_KEYS if base_env.get(k) != env.get(k)]
    sha_ok = base["model"]["sha256"] == model["sha256"]
    locked_ref = base.get("reference", {}).get("size_bytes")
    return {
        "ok": sha_ok and not diffs and not missing and compared > 0,
        "sha_ok": sha_ok, "locked_file": base["model"]["file"], "locked_sha256": base["model"]["sha256"],
        "model_file": model["file"], "model_sha256": model["sha256"], "baseline_created": base.get("created"),
        "compared": compared, "locked_total": sum(len(v) for v in base["clips"].values()),
        "diffs": diffs, "missing": missing, "inputs_changed": inputs_changed, "env_changed": env_changed,
        "reference_changed": [locked_ref, reference["size_bytes"]] if locked_ref not in (None, reference["size_bytes"]) else None,
    }


def _nbytes(t) -> int:
    """Initializer payload size without materialising it."""
    from onnx import helper

    if t.raw_data:
        return len(t.raw_data)
    n = int(np.prod(t.dims)) if t.dims else 1
    try:
        return n * np.dtype(helper.tensor_dtype_to_np_dtype(t.data_type)).itemsize
    except Exception:  # noqa: BLE001  exotic dtypes (int4...)
        return 0


def recipe_hint(path: Path, note: str | None = None) -> str:
    """One line on how the model was made: the --recipe note, the variant/desc that quantize_onnx.py
    writes to the <model>.json next to it, and what the graph itself says (quantization ops, weight
    bytes by dtype). Advisory only: it never blocks a lock."""
    parts = [note] if note else []
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        trim = meta.get("trim_pe") or {}
        parts.append(f"{meta['variant']}: {meta['desc']}"
                     + (f" (position table trimmed to {trim['max_frames']} frames)" if trim.get("trimmed") else ""))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        import onnx

        m = onnx.load(str(path), load_external_data=False)
        ops = Counter(n.op_type for n in m.graph.node)
        qops = ", ".join(f"{k}x{v}" for k, v in sorted(ops.items())
                         if any(s in k for s in ("Integer", "Quantize", "QLinear", "NBits")))
        weights = Counter()
        for t in m.graph.initializer:
            weights[onnx.TensorProto.DataType.Name(t.data_type)] += _nbytes(t)
        wtxt = ", ".join(f"{k} {v / 1e6:.0f} MB" for k, v in weights.most_common() if v >= 5e5)
        parts.append(f"graph: {qops or 'no quantized ops'}; weights {wtxt}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"graph not inspected ({type(e).__name__}: {e})")
    return " | ".join(parts)


def make_baseline(report: dict, rows: list[ClipResult], recipe: str, reference: dict) -> dict:
    clips = {}
    for split in SPLITS:
        chosen = [r for r in rows if r.split == split][: FIXED[split]]
        bad = [r for r in chosen if r.error]
        if bad or len(chosen) < FIXED[split]:
            raise SetupError(f"cannot lock: {split} fixed subset has "
                             f"{'a failed clip: ' + bad[0].id + ' ' + bad[0].error if bad else 'too few clips'}")
        clips[split] = {r.id: {"text": r.text_q, "fp32": r.text_fp32, "input_sha256": r.input_sha256}
                        for r in chosen}
    keep = ("n", "wer_fp32_pct", "wer_q_pct", "wer_delta_pts", "agree_mean", "agree_min", "text_diff", "empty_regress")
    return {
        "schema": SCHEMA,
        "created": report["created"],
        "model": {k: report["model"][k] for k in ("file", "sha256", "size_bytes", "size_mb")},
        "reference": {k: reference[k] for k in ("file", "sha256", "size_bytes")},
        "recipe_hint": recipe,
        "env": {k: v for k, v in report["env"].items() if k not in ("os", "python")},
        "locked_run": {s: report["run"][s] for s in SPLITS},
        "fixed_subset": FIXED,
        "limits": {"max_size_mb": MAX_SIZE_MB, "max_wer_delta_pts": MAX_WER_DELTA_PTS,
                   "min_mean_agreement": MIN_MEAN_AGREEMENT},
        "metrics_at_lock": {"size_mb": report["model"]["size_mb"],
                            **{s: {k: round(v[k], 4) if isinstance(v[k], float) else v[k] for k in keep}
                               for s, v in report["sets"].items()}},
        "clips": clips,
    }


def dump_baseline(b: dict) -> str:
    """JSON with one clip per line, so a re-lock reads as a clean line diff in review."""
    head = json.dumps({k: v for k, v in b.items() if k != "clips"}, indent=2, ensure_ascii=False)
    blocks = []
    for split, clips in b["clips"].items():
        lines = ",\n".join(f"      {json.dumps(cid)}: {json.dumps(rec, ensure_ascii=False)}" for cid, rec in clips.items())
        blocks.append(f'    {json.dumps(split)}: {{\n{lines}\n    }}')
    return head[:-2] + ',\n  "clips": {\n' + ",\n".join(blocks) + "\n  }\n}\n"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ───────────────────────────── the run ─────────────────────────────

def evaluate(o: Options) -> dict:
    """Run both models over the clips; return the report dict (see render_markdown / report['ok']).
    Raises SetupError for missing inputs. Prints progress to stderr unless o.quiet."""
    t_start = time.perf_counter()
    say = (lambda *_: None) if o.quiet else (lambda *a: print(*a, file=sys.stderr, flush=True))
    if why := missing_inputs(o):
        raise SetupError(why)
    tokens = json.loads((o.tokens or o.reference.parent / "tokens.json").read_text())

    clips = {"lrs3": load_lrs3(o.lrs3), "raw": bench.clips_from_dir(RAW_DIR, o.raw) if o.raw > 0 else []}
    if o.update_baseline:
        for split in SPLITS:
            if len(clips[split]) < FIXED[split]:
                raise SetupError(f"--update-baseline locks the first {FIXED[split]} {split} clips but only "
                                 f"{len(clips[split])} are available")
    if not any(clips.values()):
        raise SetupError("no clips selected")

    from lipread.model import collapse_ctc, ids_to_text
    from lipread.preprocess import precropped_patches, to_model_input
    from lipread.video import load_video_25fps

    model = file_info(o.model)
    reference = file_info(o.reference, sha=o.update_baseline)  # hashing 775 MB only when it gets recorded
    say(f"models: {model['file']} ({model['size_mb']:.0f} MB) vs fp32 {reference['file']} ({reference['size_mb']:.0f} MB)")
    t = time.perf_counter()
    sess = {"fp32": open_session(o.reference, o.threads), "quant": open_session(o.model, o.threads)}
    load_s = time.perf_counter() - t
    contract = {tag: io_problems(s, len(tokens)) for tag, s in sess.items()}

    cropper = None

    def preprocess(clip: bench.Clip) -> np.ndarray:  # the same path bench.OnnxBackend takes
        nonlocal cropper
        if clip.path:
            from lipread.preprocess import MouthCropper

            cropper = cropper or MouthCropper()
            patches = cropper.crop(load_video_25fps(clip.path))
        else:
            patches = precropped_patches(clip.crops)
        return to_model_input(patches).unsqueeze(0).numpy()  # (1, 1, T, 88, 88) float32

    def run(tag: str, x: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        out = sess[tag].run(None, {IN_NAME: x})[0]
        return out, (time.perf_counter() - t0) * 1000

    work = [(split, c) for split in SPLITS for c in clips[split]]
    rows: list[ClipResult] = []
    warmed = o.fast  # --fast (smoke) is time-boxed and only indicative on latency, so it skips the warmup
    pieces: list[np.ndarray] = []  # real inputs, stitched into one LONG_FRAMES probe after the loop
    for k, (split, clip) in enumerate(work):
        row = ClipResult(clip.id, split, clip.ref)
        rows.append(row)
        try:
            x = preprocess(clip)  # once: both models get this exact array
        except Exception as e:  # noqa: BLE001  NoFaceError, undecodable video ...
            row.error = f"preprocess: {type(e).__name__}: {' '.join(str(e).split())[:160]}"
            continue
        row.frames, row.input_sha256 = int(x.shape[2]), hashlib.sha256(x.tobytes()).hexdigest()[:16]
        if not o.fast and sum(p.shape[2] for p in pieces) < LONG_FRAMES:
            pieces.append(x)
        try:
            if not warmed:  # untimed: arena allocation, first-use kernel setup
                for tag in sess:
                    run(tag, x)
                warmed = True
            out = {}
            # Alternate who goes first so load from other processes and cache warmth hit both alike.
            for tag in (("fp32", "quant") if k % 2 == 0 else ("quant", "fp32")):
                out[tag] = run(tag, x)
        except Exception as e:  # noqa: BLE001
            row.error = f"onnxruntime run: {type(e).__name__}: {' '.join(str(e).split())[:160]}"
            continue
        (lp_ref, row.ms_fp32), (lp_q, row.ms_q) = out["fp32"], out["quant"]
        nonfinite = [tag for tag, (lp, _) in out.items() if not np.isfinite(lp).all()]
        if lp_ref.shape != lp_q.shape:
            row.error = f"output shape {lp_q.shape} != fp32 {lp_ref.shape}"
        elif nonfinite:
            row.error = f"non-finite log_probs from {', '.join(nonfinite)}"
        else:
            best_ref, best_q = lp_ref.argmax(-1), lp_q.argmax(-1)
            row.agree = float((best_ref == best_q).mean())
            row.text_fp32 = ids_to_text(collapse_ctc(best_ref.tolist()), tokens)
            row.text_q = ids_to_text(collapse_ctc(best_q.tolist()), tokens)
        if (k + 1) % 20 == 0 or k + 1 == len(work):
            say(f"  {k + 1}/{len(work)} clips ({time.perf_counter() - t_start:.0f} s)")

    long_probe = None
    if pieces:  # not an accuracy test: the position table / dynamic shapes must survive a maximum-length utterance
        xl = np.concatenate(pieces, axis=2)
        xl = np.ascontiguousarray(np.tile(xl, (1, 1, -(-LONG_FRAMES // xl.shape[2]), 1, 1))[:, :, :LONG_FRAMES])
        long_probe = {"frames": LONG_FRAMES, "problem": None, "agree": None}
        try:
            (lp_ref, _), (lp_q, _) = run("fp32", xl), run("quant", xl)
            if lp_q.shape != (LONG_FRAMES, len(tokens)) or lp_ref.shape != lp_q.shape:
                long_probe["problem"] = f"output shape {lp_q.shape}, expected {(LONG_FRAMES, len(tokens))}"
            elif not np.isfinite(lp_q).all():
                long_probe["problem"] = "non-finite log_probs"
            else:
                long_probe["agree"] = float((lp_ref.argmax(-1) == lp_q.argmax(-1)).mean())
        except Exception as e:  # noqa: BLE001
            long_probe["problem"] = f"onnxruntime run: {type(e).__name__}: {' '.join(str(e).split())[:160]}"

    ran = {s: len(clips[s]) for s in SPLITS}
    sets = {s: summarize_set([r for r in rows if r.split == s]) for s in SPLITS if ran[s]}
    env = env_info(o.threads)
    lock = None
    if o.baseline and Path(o.baseline).is_file():
        try:
            lock = compare_lock(load_baseline(o.baseline), model, reference, rows, env, ran)
        except SetupError as e:
            if not o.update_baseline:  # but a stale or damaged baseline must not block re-locking
                raise
            say(f"ignoring the existing baseline: {e}")
    gates = build_gates(model, contract, sets, lock, enforce_lock=not o.update_baseline, long_probe=long_probe)
    fast_gates = []
    if o.update_baseline:  # smoke.sh runs the --fast subset: don't lock a model that would turn it red
        fast_sets = {s: summarize_set([r for r in rows if r.split == s][: FAST[s]]) for s in SPLITS}
        fast_gates = [g for g in build_gates(model, contract, fast_sets, None) if g["scope"] in SPLITS]
    return {
        "schema": SCHEMA,
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ok": not any(g["status"] == "FAIL" for g in gates),
        "mode": "fast" if o.fast else "full" if (o.lrs3, o.raw) == (FULL["lrs3"], FULL["raw"]) else "custom",
        "model": model, "reference": reference, "env": env,
        "run": {**ran, "seconds": round(time.perf_counter() - t_start, 1), "session_load_s": round(load_s, 2),
                "warmup_runs": 1, "greedy": True},
        "sets": sets, "long_input": long_probe, "gates": gates, "fast_gates": fast_gates, "lock": lock,
        "rows": [asdict(r) for r in rows],
    }


# ───────────────────────────── reporting ─────────────────────────────

def table(headers: list[str], rows: list[list]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    line = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"  # noqa: E731
    return "\n".join([line(headers), "|" + "|".join("-" * (x + 2) for x in w) + "|", *map(line, cells)])


def word_diff(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        if tag == "replace":
            ops.append(f"{' '.join(aw[i1:i2])}→{' '.join(bw[j1:j2])}")
        elif tag == "delete":
            ops.append(f"-{' '.join(aw[i1:i2])}")
        elif tag == "insert":
            ops.append(f"+{' '.join(bw[j1:j2])}")
    return ", ".join(ops)


def lock_notes(lock: dict) -> list[str]:
    """What moved since the lock besides the texts: the usual suspects when a text does change."""
    out = []
    if lock["sha_ok"] and lock["model_file"] != lock["locked_file"]:
        out.append(f"- model file renamed: {lock['locked_file']} -> {lock['model_file']} (sha256 identical)")
    if lock["env_changed"]:
        out.append("- environment differs from the lock: " + "; ".join(f"{k} {a} -> {b}" for k, a, b in lock["env_changed"]))
    if lock["inputs_changed"]:
        ids = lock["inputs_changed"]
        out.append(f"- model inputs differ from the lock for {len(ids)} clip(s) ({', '.join(ids[:5])}"
                   f"{', ...' if len(ids) > 5 else ''}): preprocessing, video decoding or the eval data changed")
    if lock["reference_changed"]:
        a, b = lock["reference_changed"]
        out.append(f"- the fp32 reference differs from the lock ({a / 1e6:.1f} -> {b / 1e6:.1f} MB): "
                   "agreement and WER deltas now measure against a different fp32")
    return out


def lock_detail(lock: dict, heading: str) -> list[str]:
    out = [f"**{heading}** (locked `{lock['locked_file']}` at {lock.get('baseline_created') or '?'})", ""]
    if not lock["sha_ok"]:
        out.append(f"- model sha256 changed: locked `{lock['locked_sha256'][:12]}` ({lock['locked_file']}) "
                   f"-> now `{lock['model_sha256'][:12]}` ({lock['model_file']})")
    if lock["diffs"]:
        out += [f"- {len(lock['diffs'])}/{lock['compared']} locked clip texts changed:", "", "```"]
        for d in lock["diffs"][:MAX_SHOWN]:
            out += [d["id"], f"  ref   : {d['ref']}", f"  locked: {d['locked'] or '(empty)'}",
                    f"  now   : {d['now'] or '(empty)'}",
                    f"  change: {word_diff(d['locked'], d['now']) or '(whitespace)'}"]
        if len(lock["diffs"]) > MAX_SHOWN:
            out.append(f"... and {len(lock['diffs']) - MAX_SHOWN} more")
        out += ["```", ""]
    if lock["missing"]:
        out.append(f"- locked clips not found in this run: {', '.join(lock['missing'])} (eval data changed?)")
    return out + lock_notes(lock)


def render_failures(r: dict, lock_fails: bool = True) -> str:
    """What failed, as markdown: the failing gate rows, the --fast rows (when locking), then the lock diff."""
    cols = lambda gs: table(["gate", "scope", "value", "limit"], [[g["name"], g["scope"], g["value"], g["limit"]] for g in gs])  # noqa: E731
    out = []
    if failed := [g for g in r["gates"] if g["status"] == "FAIL"]:
        out += ["**FAILED gates**", "", cols(failed)]
    if fast_failed := [g for g in r["fast_gates"] if g["status"] == "FAIL"]:
        out += ["", f"**the --fast subset (first {FAST['lrs3']} LRS3 + {FAST['raw']} raw clips, what smoke.sh runs) "
                    "would FAIL**", "", cols(fast_failed)]
    if lock_fails and r["lock"] and not r["lock"]["ok"]:
        out += ["", *lock_detail(r["lock"], "baseline lock FAILED"), "",
                "Intended change? Review the above and a full run (no --fast), then re-run with "
                "`--update-baseline` and commit ml/tests/quantized_baseline.json."]
    return "\n".join(out)


def render_markdown(r: dict, update: bool = False) -> str:
    m, ref, run, env = r["model"], r["reference"], r["run"], r["env"]
    counts = " + ".join(f"{run[s]} {s}" for s in SPLITS if run[s])
    out = [f"## Quantized regression: `{m['file']}` vs fp32 `{ref['file']}`", "",
           f"{counts} clips ({r['mode']}) · greedy CTC · identical inputs · onnxruntime {env['onnxruntime']} CPU "
           f"({env['threads']} threads) · {run['seconds']:.0f} s · {r['created']}", ""]
    rows = []
    for split, s in r["sets"].items():
        if not s["n_ok"]:
            rows.append([split, s["n"], "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        speed = s["lat_fp32_p50"] / s["lat_q_p50"] if s["lat_q_p50"] else float("nan")
        rows.append([split, s["n"], f"{s['wer_fp32_pct']:.1f}%", f"{s['wer_q_pct']:.1f}%", f"{s['wer_delta_pts']:+.2f}",
                     f"{s['agree_mean']:.3f} / {s['agree_min']:.3f}", s["text_diff"], s["empty_regress"],
                     f"{s['lat_fp32_p50']:.0f} / {s['lat_fp32_p95']:.0f}", f"{s['lat_q_p50']:.0f} / {s['lat_q_p95']:.0f}",
                     f"{speed:.2f}x"])
    out += [table(["set", "n", "WER fp32", "WER quant", "Δ pts", "argmax agree mean / min", "text ≠ fp32",
                   "quant empty, fp32 not", "fp32 ms p50 / p95", "quant ms p50 / p95", "p50 speedup"], rows), ""]
    out += [table(["model", "file", "size MB", "sha256"],
                  [["fp32", ref["file"], f"{ref['size_mb']:.1f}", (ref["sha256"] or "-")[:12]],
                   ["quant", m["file"], f"{m['size_mb']:.1f}", m["sha256"][:12]]]), ""]
    out += [table(["gate", "scope", "value", "limit", "status"],
                  [[g["name"], g["scope"], g["value"], g["limit"], g["status"]] for g in r["gates"]]), ""]
    st = Counter(g["status"] for g in r["gates"])
    verdict = "PASS" if r["ok"] else "FAIL"
    detail = ", ".join(f"{st[k]} {k.lower()}" for k in ("PASS", "FAIL", "SKIP", "INFO") if st[k])
    out.append(f"**gates: {verdict}** ({detail})")
    lock = r["lock"]
    if failures := render_failures(r, lock_fails=not update):
        out += ["", failures]
    if lock and update and not lock["ok"]:  # re-locking: show what the new baseline changes
        out += ["", *lock_detail(lock, "changes vs the previous baseline (about to be replaced)")]
    elif lock and lock["ok"] and (notes := lock_notes(lock)):
        out += ["", "**the lock holds, but things moved**", "", *notes]
    return "\n".join(out)


def one_line(r: dict) -> str:
    parts = [f"{r['model']['file']} {r['model']['size_mb']:.0f} MB"]
    for split, s in r["sets"].items():
        if s["n_ok"]:
            parts.append(f"{split} n={s['n']} WER {s['wer_fp32_pct']:.1f}->{s['wer_q_pct']:.1f}% "
                         f"agree {s['agree_mean']:.3f}")
    lock = r["lock"]
    parts.append(f"lock {lock['compared'] - len(lock['diffs'])}/{lock['compared']} identical" if lock else "no lock")
    return " · ".join(parts)


# ───────────────────────────── CLI ─────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="quantized .onnx (default: newest artifacts/lipread_ctc.<variant>.onnx)")
    ap.add_argument("--reference", type=Path, default=FP32, help="fp32 reference (default: artifacts/lipread_ctc.onnx)")
    ap.add_argument("--tokens", type=Path, help="tokens.json (default: next to --reference)")
    ap.add_argument("--lrs3", type=int, help=f"LRS3 clips (default {FULL['lrs3']}; {FAST['lrs3']} with --fast)")
    ap.add_argument("--raw", type=int, help=f"raw eval clips (default {FULL['raw']}; {FAST['raw']} with --fast)")
    ap.add_argument("--fast", action="store_true", help=f"{FAST['lrs3']} LRS3 + {FAST['raw']} raw clips, for smoke")
    ap.add_argument("--baseline", type=Path, default=BASELINE, help="lock file (default: ml/tests/quantized_baseline.json)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the baseline for this model (full run; refuses if a gate fails unless --force)")
    ap.add_argument("--force", action="store_true", help="with --update-baseline: lock even though a gate fails")
    ap.add_argument("--recipe", help="free-text recipe note stored in the baseline (the graph summary is added automatically)")
    ap.add_argument("--out", type=Path, help="JSON report (default: artifacts/regress/<model-stem>[.fast].json)")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (default: ORT decides)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        model = a.model or find_quantized()
        if model is None:
            raise SetupError("no quantized model: pass --model or create artifacts/lipread_ctc.<variant>.onnx")
        base = FAST if a.fast else FULL
        o = Options(model=model, reference=a.reference, tokens=a.tokens,
                    lrs3=base["lrs3"] if a.lrs3 is None else a.lrs3, raw=base["raw"] if a.raw is None else a.raw,
                    baseline=a.baseline, fast=a.fast, update_baseline=a.update_baseline, threads=a.threads)
        if a.update_baseline and (a.fast or o.lrs3 < FIXED["lrs3"] or o.raw < FIXED["raw"]):
            raise SetupError(f"--update-baseline is a full-run operation (needs >= {FIXED['lrs3']} LRS3 and "
                             f">= {FIXED['raw']} raw clips, no --fast)")
        if a.model is None:
            n = len(list(ART.glob("lipread_ctc.*.onnx")))
            print(f"model: {rel(model)} (newest of {n} lipread_ctc.<variant>.onnx in artifacts/)", file=sys.stderr)
        report = evaluate(o)
    except SetupError as e:
        print(f"regress_quantized: {e}", file=sys.stderr)
        return 2
    except ModelLoadError as e:  # the model itself is what is broken: a failure, not a setup problem
        print(f"regress_quantized: FAIL {e}", file=sys.stderr)
        return 1

    out = a.out or ART / "regress" / f"{model.stem}{'.fast' if a.fast else ''}.json"
    write_text_atomic(out, json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(render_markdown(report, update=a.update_baseline))
    print(f"\nreport: {rel(out)}")

    if not a.update_baseline:
        return 0 if report["ok"] else 1

    blockers = [g for g in report["gates"] + report["fast_gates"] if g["status"] == "FAIL"]
    if blockers and not a.force:
        print("\nREFUSING to write the baseline: a gate fails (fix the model, or --force to lock it anyway)")
        return 1
    try:
        reco = recipe_hint(model, a.recipe)
        baseline = make_baseline(report, [ClipResult(**row) for row in report["rows"]], reco, report["reference"])
    except SetupError as e:
        print(f"regress_quantized: {e}", file=sys.stderr)
        return 2
    write_text_atomic(a.baseline, dump_baseline(baseline))
    n = sum(len(v) for v in baseline["clips"].values())
    print(f"\nbaseline written: {rel(a.baseline)}  ({model.name}, sha256 {baseline['model']['sha256'][:12]}, "
          f"{n} locked clips{', LOCKED DESPITE FAILING GATES (--force)' if blockers else ''})\n"
          f"recipe: {reco}\nnext: review the file and commit it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
