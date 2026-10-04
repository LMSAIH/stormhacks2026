"""Fast ML sanity checks for ../smoke.sh. Prints one `PASS|FAIL|SKIP <name>` line per check.

Exit code = number of failures. A real-face clip at ml/data/smoke/face.mp4 (or $ML_SMOKE_CLIP)
enables the end-to-end check (any ~3 s webcam clip of a face). The quantized-model regression check
needs a quantized model plus ml/tests/quantized_baseline.json ($ML_QUANT_MODEL / $ML_QUANT_BASELINE
override both) and SKIPs without them.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("LIPREAD_CKPT_DIR", ML / "checkpoints")) / "LRS3_V_WER19.1" / "model.pth"
CLIP = Path(os.environ.get("ML_SMOKE_CLIP", ML / "data" / "smoke" / "face.mp4"))

results: list[tuple[str, str, str]] = []


class Skip(Exception):
    pass


def check(name):
    def deco(fn):
        t = time.perf_counter()
        try:
            detail = fn() or ""
            results.append(("PASS", name, f"{detail} ({time.perf_counter() - t:.1f}s)"))
        except Skip as s:
            results.append(("SKIP", name, str(s)))
        except Exception as e:  # noqa: BLE001
            results.append(("FAIL", name, f"{type(e).__name__}: {e}"))
            traceback.print_exc(file=sys.stderr)
        print(*results[-1], flush=True)
        return fn
    return deco


def noise_clip(path: Path, n: int, fps: float) -> None:
    from lipread.video import write_video
    rng = np.random.default_rng(0)
    write_video(path, rng.integers(0, 255, (n, 120, 160, 3), dtype=np.uint8), fps=fps)


def gray_clip(path: Path, n: int, size=(120, 160)) -> None:
    """Flat gray frames: a deterministic no-face fixture (random noise can fool BlazeFace)."""
    from lipread.video import write_video
    write_video(path, np.full((n, *size, 3), 128, dtype=np.uint8))


@check("imports")
def _():
    import espnet  # noqa: F401  vendored Chaplin copy
    import mediapipe as mp
    import torch

    import lipread.serve.app  # noqa: F401
    assert hasattr(mp.solutions, "face_detection"), "mediapipe lost the legacy solutions API"
    if os.environ.get("SMOKE_REQUIRE_CUDA") == "1":
        assert torch.cuda.is_available(), "SMOKE_REQUIRE_CUDA=1 but torch.cuda.is_available() is False"
    gpu = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    return f"torch {torch.__version__} cuda={torch.cuda.is_available()}{gpu}"


@check("video io + 30→25 fps resample")
def _():
    from lipread.video import load_video, resample_fps
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "n.mp4"
        noise_clip(p, 60, 30.0)  # 2 s
        frames, fps = load_video(p)
    out = resample_fps(frames, fps)
    assert frames.shape[1:] == (120, 160, 3), frames.shape
    assert len(out) == 50, len(out)
    return f"{len(frames)}@{fps:.0f} → {len(out)}@25"


@check("preprocess rejects faceless video (flat gray + noise)")
def _():
    from lipread.preprocess import MouthCropper, NoFaceError
    from lipread.video import load_video_25fps
    cropper = MouthCropper()
    for name, make in (("gray", lambda p: gray_clip(p, 25)), ("noise", lambda p: noise_clip(p, 25, 25.0))):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"{name}.mp4"
            make(p)
            frames = load_video_25fps(p)
        try:
            cropper.crop(frames)
        except NoFaceError:
            continue
        raise AssertionError(f"expected NoFaceError on {name} (face-coverage gate)")
    return "NoFaceError x2"


reader = None


@check("model load + greedy decode (random input)")
def _():
    global reader
    if not CKPT.is_file():
        raise Skip(f"no checkpoint at {CKPT} (run scripts/download_checkpoints.sh)")
    import torch

    from lipread.model import LipReader
    reader = LipReader(use_lm=False, beam_size=4)
    logp = reader.ctc_log_probs(torch.randn(1, 50, 88, 88))
    assert logp.shape == (50, len(reader.token_list)), tuple(logp.shape)
    assert torch.isfinite(logp).all()
    reader.greedy(torch.randn(1, 50, 88, 88))
    return f"vocab={len(reader.token_list)} device={reader.device}"


@check("onnx export parity (if exported)")
def _():
    onnx_path = ML / "artifacts" / "lipread_ctc.onnx"
    if not onnx_path.is_file():
        raise Skip("no artifacts/lipread_ctc.onnx (run scripts/export_onnx.py)")
    if reader is None:
        raise Skip("model not loaded")
    import onnxruntime as ort
    import torch
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 1, 37, 88, 88)
    out = sess.run(None, {"video": x.numpy()})[0]
    ref = reader.ctc_log_probs(x[0]).cpu().numpy()  # ctc_log_probs adds the batch dim
    agree = float((out.argmax(-1) == ref.argmax(-1)).mean())
    assert agree >= 0.99, f"argmax agreement {agree:.1%}"
    return f"agree={agree:.0%} max|Δ|={float(np.abs(out - ref).max()):.1e}"


@check("service /health + /lipread error path")
def _():
    from fastapi.testclient import TestClient

    from lipread.serve.app import app
    c = TestClient(app)
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok", h.text
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.mp4"
        gray_clip(p, 25)
        r = c.post("/lipread", files={"file": ("g.mp4", p.read_bytes(), "video/mp4")})
        assert r.status_code == 422 and r.json()["detail"]["error"] == "no_face_detected", r.text
        if not CKPT.is_file():
            return "422 no_face_detected (precropped path skipped: no checkpoint)"
        q = Path(d) / "c.mp4"
        gray_clip(q, 30, size=(96, 96))
        r = c.post("/lipread", files={"file": ("c.mp4", q.read_bytes(), "video/mp4")},
                   data={"precropped": "true", "correct": "false"})
    assert r.status_code == 200, r.text
    lat = r.json()["latency_ms"]
    assert set(lat) >= {"load", "crop", "vsr", "total"}, lat
    return f"422 no_face_detected; precropped 200 vsr={lat['vsr']:.0f}ms"


@check("service /lipread/crops (raw + gzip, 96 + 88, rejections)")
def _():
    import gzip

    import torch
    from fastapi.testclient import TestClient

    import lipread.serve.app as service
    from lipread.preprocess import to_model_input
    c = TestClient(service.app)
    p96 = np.random.default_rng(1).integers(0, 256, (30, 96, 96), dtype=np.uint8)
    p88 = np.ascontiguousarray(p96[:, 4:92, 4:92])
    # 88 is the centre crop the model sees, so both sizes must give bit-identical model input.
    assert torch.equal(to_model_input(p96), to_model_input(p88)), "88 vs 96 model input differs"

    def post(body: bytes, t: int = 30, size: int = 96, encoding: str | None = None, **query):
        headers = {"Content-Type": "application/octet-stream"}
        if encoding:
            headers["Content-Encoding"] = encoding
        params = {"t": t, "h": size, "w": size, "decode": "greedy", **query}
        return c.post("/lipread/crops", params=params, content=body, headers=headers)

    def rejects(r, status: int, error: str) -> None:
        assert r.status_code == status and r.json()["detail"]["error"] == error, \
            f"want {status} {error}, got {r.status_code} {r.text[:200]}"

    bomb = gzip.compress(bytes(20_000_000))  # 20 MB of zeros in ~20 KB
    assert len(service._gunzip(bomb, 1000)) == 1001, "gunzip must stop right after the size limit"
    rejects(post(bomb, encoding="gzip"), 422, "body_size_mismatch")
    rejects(post(p96.tobytes(), size=64), 422, "bad_shape")
    rejects(post(p96[:12].tobytes(), t=12), 422, "clip_too_short")
    rejects(post(p96[:29].tobytes()), 422, "body_size_mismatch")
    rejects(post(b"not gzip", encoding="gzip"), 422, "bad_gzip")
    rejects(post(p96.tobytes(), encoding="br"), 415, "unsupported_encoding")
    rejects(post(bytes(service.MAX_CROPS_BODY + 1), t=250), 413, "body_too_large")
    params = {p["name"]: p["schema"].get("default") for p in
              service.app.openapi()["paths"]["/lipread/crops"]["post"]["parameters"]}
    assert params["decode"] == "beam" and params["h"] == params["w"] == 96 and params["correct"] is False, params
    # Browsers preflight this request (Content-Encoding is not a CORS-safelisted header).
    pre = c.options("/lipread/crops", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-encoding,content-type"})
    assert pre.status_code == 200 and "content-encoding" in pre.headers.get("access-control-allow-headers", ""), \
        f"CORS preflight: {pre.status_code} {dict(pre.headers)}"
    if not CKPT.is_file():
        return "rejections + CORS ok (decode paths skipped: no checkpoint)"

    ref = service.reader().greedy(to_model_input(p96))  # in-process: the endpoint must be lossless
    gz = gzip.compress(p96.tobytes())
    for name, r in (("raw96", post(p96.tobytes())), ("gzip96", post(gz, encoding="gzip")),
                    ("raw88", post(p88.tobytes(), size=88))):
        assert r.status_code == 200, f"{name}: {r.status_code} {r.text[:200]}"
        j = r.json()
        assert j["frames"] == 30 and set(j["latency_ms"]) == {"load", "crop", "vsr", "correct", "total"}, j
        assert j["alternatives"] == [], f"{name}: greedy must not return beam alternatives: {j['alternatives']}"
        conf_ok = (j["confidence"] is None) == (ref.confidence is None) and (
            ref.confidence is None or abs(j["confidence"] - ref.confidence) < 1e-6)
        assert j["raw_text"] == ref.text and conf_ok, f"{name} {j['raw_text']!r}/{j['confidence']} " \
                                                      f"!= in-process {ref.text!r}/{ref.confidence}"
    return f"7 rejections + CORS ok; raw/gzip/88 == in-process greedy, vsr={j['latency_ms']['vsr']:.0f}ms"


@check("end-to-end on real face clip")
def _():
    if not CLIP.is_file():
        raise Skip(f"no clip at {CLIP} (save a ~3 s webcam clip of your face there)")
    if reader is None:
        raise Skip("model not loaded")
    import gzip

    from fastapi.testclient import TestClient

    import lipread.serve.app as service
    from lipread.preprocess import MouthCropper, to_model_input
    from lipread.video import load_video_25fps
    crops = MouthCropper().crop(load_video_25fps(CLIP))
    x = to_model_input(crops)
    assert x.shape[0] == 1 and x.shape[2:] == (88, 88), tuple(x.shape)
    text = reader.greedy(x).text
    # Accuracy-mode transport: the same crops, gzipped like the browser sends them.
    body = gzip.compress(np.ascontiguousarray(crops).tobytes())
    r = TestClient(service.app).post("/lipread/crops", params={"t": len(crops), "decode": "greedy"},
                                     content=body, headers={"Content-Encoding": "gzip"})
    assert r.status_code == 200 and r.json()["raw_text"] == text, \
        f"/lipread/crops: {r.status_code} {r.text[:200]} vs in-process {text!r}"
    return f"{text!r} (/lipread/crops agrees, {len(body) / 1024:.0f} KB gzip)"


@check("service /training-pairs (opt-in clips)")
def _():
    import gzip
    import json

    from fastapi.testclient import TestClient

    import lipread.serve.app as service

    crops = np.random.default_rng(2).integers(0, 256, (30, 96, 96), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("LIPREAD_PAIRS_DIR"), os.environ.pop("LIPREAD_PAIRS_REPO", None)
        os.environ["LIPREAD_PAIRS_DIR"] = tmp
        try:
            c = TestClient(service.app)
            post = lambda body, **q: c.post(  # noqa: E731
                "/training-pairs", params={"t": 30, "text": " i think  i have ", "source": "picked", **q},
                content=body, headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip"})
            r = post(gzip.compress(crops.tobytes()))
            assert r.status_code == 200, r.text[:200]
            j = r.json()
            assert j["text"] == "I THINK I HAVE" and j["uploading_to"] is None, j
            files = sorted(p.suffix for p in Path(tmp).iterdir())
            assert files == [".json", ".npz", ".txt"], files
            saved = np.load(Path(tmp) / f"{j['id']}.npz")["crops"]
            assert np.array_equal(saved, crops), "stored crops differ from the upload"
            meta = json.loads((Path(tmp) / f"{j['id']}.json").read_text())
            assert meta["source"] == "picked" and meta["frames"] == 30, meta
            assert post(gzip.compress(crops.tobytes()), source="bogus").status_code == 422
            assert post(gzip.compress(crops[:29].tobytes())).status_code == 422  # size mismatch
        finally:
            if old[0] is None:
                os.environ.pop("LIPREAD_PAIRS_DIR", None)
            else:
                os.environ["LIPREAD_PAIRS_DIR"] = old[0]
            if old[1] is not None:
                os.environ["LIPREAD_PAIRS_REPO"] = old[1]
    return f"pair {j['id']} stored (npz+txt+json, crops round-trip exactly); bad source/size rejected"


@check("quantized regression (fast)")
def _():
    # scripts/regress_quantized.py --fast, in-process: 5 LRS3 + 2 raw clips through fp32 and the quantized
    # ONNX; asserts the gates and that its greedy texts + sha256 still match tests/quantized_baseline.json.
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import regress_quantized as rq

    baseline = Path(os.environ.get("ML_QUANT_BASELINE", rq.BASELINE))
    model = Path(os.environ["ML_QUANT_MODEL"]) if os.environ.get("ML_QUANT_MODEL") else rq.locked_or_newest(baseline)
    if model is None:
        raise Skip("no quantized model (artifacts/lipread_ctc.<variant>.onnx; build it with scripts/quantize_onnx.py)")
    if not baseline.is_file():
        raise Skip(f"no baseline at {rq.rel(baseline)} to check {model.name} against "
                   "(after reviewing a full run: scripts/regress_quantized.py --update-baseline)")
    opts = rq.Options(model=model, baseline=baseline, lrs3=rq.FAST["lrs3"], raw=rq.FAST["raw"], fast=True, quiet=True)
    if why := rq.missing_inputs(opts):
        raise Skip(why)
    report = rq.evaluate(opts)
    if not report["ok"]:
        print(rq.render_markdown(report), file=sys.stderr)
        raise AssertionError("quantized model regressed (table on stderr):\n" + rq.render_failures(report))
    return rq.one_line(report)


fails = sum(r[0] == "FAIL" for r in results)
print(f"ml checks: {sum(r[0] == 'PASS' for r in results)} pass, {fails} fail, "
      f"{sum(r[0] == 'SKIP' for r in results)} skip")
sys.exit(fails)
