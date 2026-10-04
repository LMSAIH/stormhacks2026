"""Decision logic of scripts/regress_quantized.py on synthetic results: gates, lock comparison, baseline
file. No models, no data; runs in about a second:  uv run pytest tests
(The end-to-end path is `uv run python scripts/regress_quantized.py`; see README.md.)"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import regress_quantized as rq  # noqa: E402

MODEL = {"file": "lipread_ctc.int8.onnx", "sha256": "a" * 64, "size_bytes": 200_000_000, "size_mb": 200.0}
REF = {"file": "lipread_ctc.onnx", "sha256": None, "size_bytes": 774_704_988, "size_mb": 774.7}
ENV = {"onnxruntime": "1.30.0", "cpu": "test cpu", "isa": "avx2", "threads": "default", "numpy": "1.26.4"}
CONTRACT = {"fp32": [], "quant": []}
FULL_RAN = {"lrs3": 10, "raw": 5}


def make_rows(n_lrs3: int = 10, n_raw: int = 5) -> list[rq.ClipResult]:
    rows = []
    for split, n in (("lrs3", n_lrs3), ("raw", n_raw)):
        for i in range(n):
            cid = f"lrs3-{i}" if split == "lrs3" else f"clip_{i:02d}"
            rows.append(rq.ClipResult(cid, split, f"hello world {i}", frames=50, text_fp32=f"HELLO WORLD {i}",
                                      text_q=f"HELLO WORLD {i}", agree=0.99, ms_fp32=20.0, ms_q=15.0,
                                      input_sha256=f"in{split}{i}"))
    return rows


def make_baseline(rows: list[rq.ClipResult], model: dict = MODEL) -> dict:
    sets = {s: rq.summarize_set([r for r in rows if r.split == s]) for s in rq.SPLITS}
    report = {"created": "2026-10-03T00:00:00Z", "model": model, "env": ENV, "run": {"lrs3": 100, "raw": 20},
              "sets": sets}
    return rq.make_baseline(report, rows, "unit test", REF)


def lock_of(rows, base, model=MODEL, env=ENV, ran=FULL_RAN, reference=REF) -> dict:
    return rq.compare_lock(base, model, reference, rows, env, ran)


def gate(gates: list[dict], name: str, scope: str | None = None) -> dict:
    return next(g for g in gates if g["name"] == name and (scope is None or g["scope"] == scope))


# ── the lock ──────────────────────────────────────────────────────────────

def test_lock_identical_run_passes():
    rows = make_rows()
    lock = lock_of(rows, make_baseline(rows))
    assert lock["ok"] and lock["compared"] == 15 and lock["locked_total"] == 15 and not lock["diffs"]
    assert not rq.lock_notes(lock)


def test_lock_catches_a_changed_text():
    base = make_baseline(make_rows())
    rows = make_rows()
    rows[3].text_q = "HELLO WORLD THREE"  # e.g. a re-quantized model now hears something else
    lock = lock_of(rows, base)
    assert not lock["ok"] and [d["id"] for d in lock["diffs"]] == ["lrs3-3"]
    assert lock["diffs"][0]["locked"] == "HELLO WORLD 3" and lock["diffs"][0]["now"] == "HELLO WORLD THREE"
    g = gate(rq.build_gates(MODEL, CONTRACT, {}, lock), "lock: fixed-subset texts")
    assert g["status"] == "FAIL" and "14/15 identical" in g["value"]
    text = rq.render_failures({"gates": rq.build_gates(MODEL, CONTRACT, {}, lock), "fast_gates": [], "lock": lock})
    assert "lrs3-3" in text and "3→THREE" in text and "--update-baseline" in text


def test_lock_catches_a_changed_model_even_if_texts_match():
    rows = make_rows()
    base = make_baseline(rows)
    new = {**MODEL, "sha256": "b" * 64}
    lock = lock_of(rows, base, model=new)
    assert not lock["ok"] and not lock["sha_ok"] and not lock["diffs"]
    g = gate(rq.build_gates(new, CONTRACT, {}, lock), "lock: model sha256")
    assert g["status"] == "FAIL" and "aaaaaaaa -> bbbbbbbb" in g["value"]


def test_lock_fast_run_compares_the_covered_prefix():
    base = make_baseline(make_rows())
    rows = make_rows(5, 2)
    lock = lock_of(rows, base, ran={"lrs3": 5, "raw": 2})
    assert lock["ok"] and lock["compared"] == 7 and lock["locked_total"] == 15
    rows[1].text_q = "CHANGED"
    assert [d["id"] for d in lock_of(rows, base, ran={"lrs3": 5, "raw": 2})["diffs"]] == ["lrs3-1"]
    # a change in a clip --fast does not cover is (by design) not seen by --fast
    rows = make_rows(5, 2)
    full = make_rows()
    full[8].text_q = "CHANGED"
    assert lock_of(rows, base, ran={"lrs3": 5, "raw": 2})["ok"]
    assert not lock_of(full, base)["ok"]


def test_lock_fails_when_a_covered_clip_is_missing_or_nothing_was_compared():
    base = make_baseline(make_rows())
    rows = [r for r in make_rows() if r.id != "clip_02"]
    lock = lock_of(rows, base)
    assert not lock["ok"] and lock["missing"] == ["clip_02"]
    assert not lock_of([], base, ran={})["ok"]  # a lock that verified nothing is not a pass


def test_lock_treats_a_failed_clip_as_a_change():
    base = make_baseline(make_rows())
    rows = make_rows()
    rows[0].error, rows[0].text_q = "preprocess: NoFaceError: face found in 3% of frames", ""
    lock = lock_of(rows, base)
    assert not lock["ok"] and lock["diffs"][0]["now"].startswith("<preprocess: NoFaceError")


def test_lock_notes_name_the_usual_suspects():
    rows = make_rows()
    base = make_baseline(rows)
    rows[2].input_sha256 = "different"
    lock = lock_of(rows, base, env={**ENV, "onnxruntime": "1.31.0"}, model={**MODEL, "file": "renamed.onnx"},
                   reference={**REF, "size_bytes": 1})
    notes = "\n".join(rq.lock_notes(lock))
    assert lock["ok"]  # nothing here changes a text
    for needle in ("onnxruntime 1.30.0 -> 1.31.0", "lrs3-2", "renamed", "fp32 reference differs"):
        assert needle in notes


def test_failure_report_names_every_failing_gate_and_the_fast_subset():
    failing = {"name": "size_mb", "scope": "model", "value": "317.0", "limit": "<= 220", "status": "FAIL"}
    fast = {"name": "mean argmax agreement", "scope": "raw", "value": "0.9239", "limit": ">= 0.95", "status": "FAIL"}
    text = rq.render_failures({"gates": [failing], "fast_gates": [fast], "lock": None})
    assert "size_mb" in text and "317.0" in text and "--fast subset" in text and "0.9239" in text
    assert rq.render_failures({"gates": [], "fast_gates": [], "lock": None}) == ""


def test_update_mode_lock_rows_are_informational():
    rows = make_rows()
    rows[0].text_q = "CHANGED"
    lock = lock_of(rows, make_baseline(make_rows()))
    statuses = {g["status"] for g in rq.build_gates(MODEL, CONTRACT, {}, lock, enforce_lock=False) if g["scope"] == "lock"}
    assert statuses == {"INFO"}
    assert gate(rq.build_gates(MODEL, CONTRACT, {}, None), "lock: baseline")["status"] == "SKIP"


# ── gates ─────────────────────────────────────────────────────────────────

def good_set(**kw) -> dict:
    return {"n": 100, "n_ok": 100, "errors": 0, "error_examples": [], "ref_words": 900, "wer_delta_pts": 0.5,
            "agree_mean": 0.98, "empty_regress": 0, **kw}


def status(gates: list[dict], name: str, scope: str = "lrs3") -> str:
    return gate(gates, name, scope)["status"]


def test_gates_enforce_the_spec():
    ok = rq.build_gates(MODEL, CONTRACT, {"lrs3": good_set(), "raw": good_set()}, None)
    assert not any(g["status"] == "FAIL" for g in ok)
    big = rq.build_gates({**MODEL, "size_bytes": 220_000_001}, CONTRACT, {"lrs3": good_set()}, None)
    assert status(big, "size_mb", "model") == "FAIL"
    assert status(rq.build_gates({**MODEL, "size_bytes": 220_000_000}, CONTRACT, {}, None), "size_mb", "model") == "PASS"
    bad = rq.build_gates(MODEL, CONTRACT, {"lrs3": good_set(wer_delta_pts=1.2, agree_mean=0.9499, empty_regress=1)}, None)
    assert [status(bad, n) for n in ("WER delta vs fp32", "mean argmax agreement", "quant empty, fp32 not")] == ["FAIL"] * 3
    edge = rq.build_gates(MODEL, CONTRACT, {"lrs3": good_set(wer_delta_pts=1.0, agree_mean=0.95)}, None)
    assert status(edge, "WER delta vs fp32") == status(edge, "mean argmax agreement") == "PASS"
    errs = rq.build_gates(MODEL, CONTRACT, {"lrs3": good_set(n_ok=99, errors=1, error_examples=["lrs3-4: boom"])}, None)
    assert status(errs, "clips run ok") == "FAIL"
    io = rq.build_gates(MODEL, {"fp32": [], "quant": ["inputs [('x', 'tensor(float16)', [1])]"]}, {}, None)
    assert status(io, "io contract", "model") == "FAIL"
    long_ok = {"frames": 250, "problem": None, "agree": 0.99}
    long_bad = {"frames": 250, "problem": "output shape (100, 5049), expected (250, 5049)", "agree": None}
    name = "max-length input (250 frames)"
    assert status(rq.build_gates(MODEL, CONTRACT, {}, None, long_probe=long_ok), name, "model") == "PASS"
    assert status(rq.build_gates(MODEL, CONTRACT, {}, None, long_probe=long_bad), name, "model") == "FAIL"
    assert not any(g["name"].startswith("max-length") for g in rq.build_gates(MODEL, CONTRACT, {}, None))


class _Arg:  # stands in for onnxruntime's NodeArg
    def __init__(self, name, type_, shape):
        self.name, self.type, self.shape = name, type_, shape


class _Sess:
    def __init__(self, ins, outs):
        self._ins, self._outs = ins, outs

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs


def test_io_contract_matches_what_the_frontend_feeds_and_reads():
    video = _Arg("video", "tensor(float)", [1, 1, "T", 88, 88])
    logp = _Arg("log_probs", "tensor(float)", ["T", 5049])
    assert rq.io_problems(_Sess([video], [logp]), 5049) == []
    assert rq.io_problems(_Sess([_Arg("video", "tensor(float16)", [1, 1, "T", 88, 88])], [logp]), 5049)
    assert rq.io_problems(_Sess([_Arg("input", "tensor(float)", [1, 1, "T", 88, 88])], [logp]), 5049)
    assert rq.io_problems(_Sess([_Arg("video", "tensor(float)", [1, 1, "T", 96, 96])], [logp]), 5049)
    assert rq.io_problems(_Sess([video], [_Arg("logits", "tensor(float)", ["T", 5049])]), 5049)
    assert rq.io_problems(_Sess([video], [_Arg("log_probs", "tensor(float)", ["T", 5000])]), 5049)
    assert len(rq.io_problems(_Sess([video, video], [logp, logp]), 5049)) == 2


def test_wer_tolerance_is_never_tighter_than_one_word():
    tiny = good_set(n=5, n_ok=5, ref_words=35, wer_delta_pts=2.8)  # one extra word on 35 = +2.86 pts
    assert status(rq.build_gates(MODEL, CONTRACT, {"lrs3": tiny}, None), "WER delta vs fp32") == "PASS"
    two_words = good_set(n=5, n_ok=5, ref_words=35, wer_delta_pts=5.8)
    assert status(rq.build_gates(MODEL, CONTRACT, {"lrs3": two_words}, None), "WER delta vs fp32") == "FAIL"
    # on the full sets (>= 100 words) the tolerance is exactly the specified 1.0 point
    full = rq.build_gates(MODEL, CONTRACT, {"raw": good_set(ref_words=122, wer_delta_pts=1.64)}, None)
    assert status(full, "WER delta vs fp32", "raw") == "FAIL" and "<= +1.00 pts" in gate(full, "WER delta vs fp32")["limit"]


def test_summarize_set_counts_what_the_gates_need():
    rows = make_rows(4, 0)
    rows[0].text_q = ""                      # quantized went silent, fp32 did not
    rows[1].text_fp32 = rows[1].text_q = ""  # both silent: not a regression
    rows[2].text_q, rows[2].agree = "HELLO WORLD 99", 0.5
    s = rq.summarize_set(rows)
    assert (s["n"], s["n_ok"], s["empty_regress"], s["text_diff"]) == (4, 4, 1, 2)
    assert s["agree_min"] == 0.5 and s["agree_min_clip"] == "lrs3-2" and s["wer_delta_pts"] > 0
    rows[3].error = "onnxruntime run: boom"
    s = rq.summarize_set(rows)
    assert (s["n_ok"], s["errors"]) == (3, 1) and s["error_examples"] == ["lrs3-3: onnxruntime run: boom"]


# ── baseline file, model discovery, helpers ───────────────────────────────

def test_baseline_file_roundtrips_with_one_clip_per_line(tmp_path):
    base = make_baseline(make_rows())
    text = rq.dump_baseline(base)
    assert json.loads(text) == base and text.endswith("}\n")
    assert sum(line.lstrip().startswith('"lrs3-') for line in text.splitlines()) == 10
    assert sum(line.lstrip().startswith('"clip_') for line in text.splitlines()) == 5
    path = tmp_path / "b.json"
    rq.write_text_atomic(path, text)
    assert rq.load_baseline(path) == base and not list(tmp_path.glob("*.tmp"))
    assert base["fixed_subset"] == rq.FIXED and base["model"]["sha256"] == MODEL["sha256"]
    path.write_text('{"schema": 99}')
    try:
        rq.load_baseline(path)
    except rq.SetupError as e:
        assert "--update-baseline" in str(e)
    else:
        raise AssertionError("an unknown schema must be rejected")


def test_make_baseline_refuses_a_failed_or_missing_clip():
    rows = make_rows()
    rows[4].error = "x"
    for bad in (rows, make_rows(10, 3)):
        try:
            make_baseline(bad)
        except rq.SetupError:
            continue
        raise AssertionError("cannot lock a subset with a failed or missing clip")


def test_find_quantized_picks_the_newest_variant_never_the_fp32(tmp_path):
    assert rq.find_quantized(tmp_path) is None
    (tmp_path / "lipread_ctc.onnx").write_bytes(b"fp32")
    (tmp_path / "tokens.json").write_text("[]")
    assert rq.find_quantized(tmp_path) is None
    old, new = tmp_path / "lipread_ctc.old.onnx", tmp_path / "lipread_ctc.new.onnx"
    old.write_bytes(b"1")
    new.write_bytes(b"2")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    assert rq.find_quantized(tmp_path) == new
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps({"model": {"file": old.name}}))
    assert rq.locked_or_newest(baseline, tmp_path) == old    # the locked file wins over a newer experiment
    baseline.write_text(json.dumps({"model": {"file": "gone.onnx"}}))
    assert rq.locked_or_newest(baseline, tmp_path) == new    # locked file absent: fall back to the newest


def test_file_info_hashes_the_model_and_its_sidecar(tmp_path):
    import hashlib

    m = tmp_path / "m.onnx"
    m.write_bytes(b"graph")
    info = rq.file_info(m)
    assert info["sha256"] == hashlib.sha256(b"graph").hexdigest() and info["size_bytes"] == 5
    (tmp_path / "m.onnx.data").write_bytes(b"weights")
    both = rq.file_info(m)
    assert both["size_bytes"] == 12 and both["sha256"] != info["sha256"]
    assert rq.file_info(m, sha=False)["sha256"] is None


def test_recipe_hint_combines_note_sidecar_and_graph(tmp_path):
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, save

    weights = helper.make_tensor("w", TensorProto.INT8, [1000, 1000], [0] * 1_000_000)
    graph = helper.make_graph(
        [helper.make_node("MatMulInteger", ["a", "w"], ["y"])], "g",
        [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 1000])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [1, 1000])], initializer=[weights])
    model = tmp_path / "lipread_ctc.t.onnx"
    save(helper.make_model(graph), model)
    assert "MatMulIntegerx1" in rq.recipe_hint(model) and "INT8 1 MB" in rq.recipe_hint(model)
    (tmp_path / "lipread_ctc.t.json").write_text(
        json.dumps({"variant": "t", "desc": "test recipe", "trim_pe": {"trimmed": True, "max_frames": 500}}))
    hint = rq.recipe_hint(model, "my note")
    assert hint.startswith("my note | t: test recipe (position table trimmed to 500 frames) | graph: MatMulIntegerx1")
    (tmp_path / "broken.onnx").write_bytes(b"not a model")
    assert "graph not inspected" in rq.recipe_hint(tmp_path / "broken.onnx")  # advisory: never raises


def test_word_diff_and_table():
    assert rq.word_diff("A B C", "A X C") == "B→X"
    assert rq.word_diff("A B C", "A C D") == "-B, +D"
    assert rq.word_diff("A", "A") == ""
    t = rq.table(["a", "bb"], [[1, "x"], ["long cell", "y"]]).splitlines()
    assert len({len(line) for line in t}) == 1 and t[1].startswith("|-")
