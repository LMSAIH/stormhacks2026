"""Shrink the exported speed-mode model (encoder + CTC head, `export_onnx.py`) for the browser.

    uv run python scripts/quantize_onnx.py --list                 # every recipe + what it does
    uv run python scripts/quantize_onnx.py --variant <name>       # → artifacts/lipread_ctc.<name>.onnx

fp32 `artifacts/lipread_ctc.onnx` (775 MB) is never modified. Every recipe is a composition of passes
applied to a copy of the graph:

  trim_pe     The exporter bakes the full ESPnet relative-position table [1, 9999, 768] fp32 (30.7 MB)
              into the graph although only 2T-1 rows are ever sliced out. Cut it to 2*max_frames-1 rows
              and re-base the two slice-index constants: bit-exact for T <= max_frames (default 500
              frames = 20 s @ 25 fps; the app caps utterances at 10 s = 250 frames).
  matmul      "dyn":    ORT dynamic quantisation — int8 per-channel weights + runtime-quantised uint8
                        activations (DynamicQuantizeLinear → MatMulInteger). Const-weight MatMuls only;
                        the attention QK/AV matmuls stay fp32. Fast int8 kernels in ORT CPU *and* ORT-web WASM.
              "nbits":  weight-only MatMulNBits (com.microsoft); activations stay fp32.
              "dq8":    weight-only int8 per-channel via DequantizeLinear. NOTE: ORT does not constant-fold
                        DequantizeLinear, so weights are re-dequantised on every run (slow) — kept for the log.
  conv groups pointwise (24 conformer Conv1d k=1, 85 MB fp32), resnet (20 Conv2d, 45 MB), front3d,
              depthwise (12, 1 MB) each take one mode:
              "matmul": (pointwise only) rewrite Conv1d(k=1) as Transpose→MatMul→Transpose so it gets the
                        same per-channel dynamic int8 as every other Linear;
              "fp16":   store the weight as fp16 + Cast (ORT constant-folds the Cast at load → fp32 compute,
                        half the file size, ~lossless);
              "dyn":    ConvInteger (uint8, per-tensor — ORT's only dynamic-conv option);
              "dq8":    per-channel int8 + DequantizeLinear (not folded by ORT → slow; kept for the log);
              None:     fp32.
  exclude     regexes of MatMul node names kept out of int8 (stored fp16, folded to fp32 by ORT at load).

CHOSEN RECIPE: `dyn-pw8-rn16` → artifacts/lipread_ctc.dyn-pw8-rn16.onnx (203.4 MB vs 774.7 MB fp32)
  * 110 Linear MatMuls + 24 pointwise convs (rewritten as MatMul) → int8 per-channel weights, runtime uint8
    activations: 134 MatMulInteger. The 36 attention QK/AV matmuls have no constant weight and stay fp32.
  * 20 dense ResNet/3D-front convs stored fp16 (+Cast, constant-folded by ORT → fp32 compute); the 12 tiny
    depthwise convs stay fp32; position table trimmed to T <= 500 frames (bit-exact, -27.6 MB).
  * Only standard ONNX ops (MatMulInteger, DynamicQuantizeLinear, Cast): no contrib op needed. Loads and runs on
    the vendored onnxruntime-web 1.30 WASM EP (checked under Node, 1 thread), identical at graph-opt level
    all / basic / disabled.
  * Needs no extra packages. The `nb*` recipes need `onnx-ir` (not in uv.lock):
    `uv run --with onnx-ir python scripts/quantize_onnx.py --variant nb8-pw8-rn16`.
  * T > --max-frames (500 = 20 s @ 25 fps) is NOT supported by the trimmed position table; the app caps at 250.

MEASURED (greedy CTC, ORT CPU, jiwer WER with bench.py's normalisation). Δ = absolute points vs fp32.
LRS3 = first 100 clips of data/lrs3_test/0000.parquet (811 ref words, 1 word = 0.12 pt); raw = data/raw_eval,
20 clips through our crop path (122 ref words, 1 word = 0.82 pt). agree = mean per-frame argmax agreement vs fp32.
  recipe                MB     LRS3 WER (Δ)    raw WER (Δ)     agree LRS3/raw   verdict
  fp32                  774.7  28.61           26.23           -                baseline
  dyn-mm                289.2  28.61 (+0.00)   26.23 (+0.00)   0.996 / 0.994    too big (convs stay fp32)
  dyn-mm-dqconv         192.1  29.47 (+0.86)   25.41 (-0.82)   0.991 / 0.998    rejected: DequantizeLinear is not
                                                                                folded by ORT -> 4-20x slower
  dyn-mm-dynconv        192.1  28.61 (+0.00)   26.23 (+0.00)   0.978 / 0.989    rejected: only 53% identical hyps,
                                                                                worst frame agreement 0.82, slow
  dyn-pw8-rn16 *        203.4  28.48 (-0.12)   25.41 (-0.82)   0.993 / 0.993    CHOSEN
  dyn-pw8-rn16-hp1      207.6  28.98 (+0.37)   27.05 (+0.82)   0.994 / 0.993    not better
  dyn-pw8-rn16-hp2      214.6  27.99 (-0.62)   26.23 (+0.00)   0.995 / 0.993    >210 MB, not clearly better
  nb8-pw8-rn16          224.1  28.61 (+0.00)   26.23 (+0.00)   0.999 / 1.000    most faithful, but >220 MB and
                                                                                ~1.5x slower than dyn on ORT CPU
  nb4-pw4-rn16          137.0  30.21 (+1.60)   27.87 (+1.64)   0.975 / 0.986    fails the +1.0 gate on both sets
  nb8b128-pw8-rn16      207.8  built, accuracy NOT scored (stopped before evaluation)
  nb8b128a4-pw8-rn16    207.8  built, accuracy NOT scored (stopped before evaluation)
  trim-only / pw-matmul 747.1  exactness checks of the two graph surgeries: max |dlogp| 0.0 / 8.7e-4 (fp32 noise),
                               12/12 identical hypotheses.
  300 further LRS3 clips (idx 100-399, 2253 words), paired bootstrap CI on dWER: dyn-mm -0.18 [-0.54, +0.20],
  dyn-pw8-rn16 -0.31 [-0.85, +0.22], hp2 +0.00 [-0.53, +0.53], dyn-mm-dynconv +0.75 [-0.40, +1.87].
  Never built or measured: dyn-mm-fp16conv, dyn-pw8-rn8, dq8-all (kept in the registry for completeness).

Measure any output with scripts/bench.py (`--backend onnx --onnx-path <file>`).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ML = Path(__file__).resolve().parents[1]
ART = ML / "artifacts"
DEFAULT_SRC = ART / "lipread_ctc.onnx"
DEFAULT_MAX_FRAMES = 500

CONV_GROUPS = {
    "pointwise": r"/conv_module/pointwise_cov",
    "depthwise": r"/conv_module/depthwise_conv",
    "front3d": r"/frontend/frontend3D/",
    "resnet": r"/frontend/trunk/",
}


# --------------------------------------------------------------------------- recipes
@dataclass
class Recipe:
    desc: str
    matmul: dict | None = None                       # {"mode": "dyn"|"nbits"|"dq8", ...}
    conv: dict[str, str | None] = field(default_factory=dict)  # group → mode (see module docstring)
    exclude: list[str] = field(default_factory=list)  # regexes of MatMul node names kept out of int8
    exclude_fp16: bool = True    # …stored as fp16 (+Cast, folded by ORT) instead of fp32 (saves half the bytes)
    trim_pe: bool = True


VARIANTS: dict[str, Recipe] = {
    "trim-only": Recipe("fp32 weights, only the position table trimmed (exactness / size reference)"),
    "pw-matmul": Recipe("fp32, pointwise convs rewritten as MatMul (exactness check of the rewrite)",
                        conv={"pointwise": "matmul"}),
    "dyn-mm": Recipe("dynamic int8 MatMuls; every Conv fp32", matmul={"mode": "dyn"}),
    "dyn-mm-dqconv": Recipe(
        "dynamic int8 MatMuls + per-channel int8 DequantizeLinear convs (slow: DQ not folded)",
        matmul={"mode": "dyn"}, conv={"pointwise": "dq8", "resnet": "dq8", "front3d": "dq8"}),
    "dyn-mm-dynconv": Recipe(
        "dynamic int8 MatMuls + ConvInteger (uint8 per-tensor) for all dense convs",
        matmul={"mode": "dyn"}, conv={"pointwise": "dyn", "resnet": "dyn", "front3d": "dyn"}),
    "dyn-mm-fp16conv": Recipe(
        "[never built] dynamic int8 MatMuls + all dense convs fp16 (safe reference, ~224 MB)",
        matmul={"mode": "dyn"}, conv={"pointwise": "fp16", "resnet": "fp16", "front3d": "fp16"}),
    "dyn-pw8-rn16": Recipe(
        "[CHOSEN] dynamic int8 MatMuls incl. pointwise-as-MatMul; ResNet+3D-front convs fp16 (203 MB)",
        matmul={"mode": "dyn"}, conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"}),
    "dyn-pw8-rn8": Recipe(
        "[never built] dynamic int8 MatMuls incl. pointwise-as-MatMul; ResNet/3D convs ConvInteger (smallest)",
        matmul={"mode": "dyn"}, conv={"pointwise": "matmul", "resnet": "dyn", "front3d": "dyn"}),
    "dyn-pw8-rn16-hp1": Recipe(
        "dyn-pw8-rn16 but ctc_lo + embed.0 (logit-critical) kept at fp16-stored fp32 compute",
        matmul={"mode": "dyn"}, conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"},
        exclude=[r"^/ctc_lo/MatMul$", r"/encoder/embed/embed\.0/MatMul$"]),
    "dyn-pw8-rn16-hp2": Recipe(
        "hp1 + the 12 linear_pos (position-attention) projections kept at fp16-stored fp32 compute",
        matmul={"mode": "dyn"}, conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"},
        exclude=[r"^/ctc_lo/MatMul$", r"/encoder/embed/embed\.0/MatMul$", r"/self_attn/linear_pos/MatMul$"]),
    "dq8-all": Recipe(
        "[never built] weight-only int8 per-channel everywhere via DequantizeLinear (slow: DQ not folded)",
        matmul={"mode": "dq8"}, conv={"pointwise": "dq8", "resnet": "dq8", "front3d": "dq8"}),
    "nb8-pw8-rn16": Recipe(
        "MatMulNBits 8-bit (block 32, symmetric) incl. pointwise-as-MatMul; ResNet fp16",
        matmul={"mode": "nbits", "bits": 8, "block": 32, "sym": True, "acc": 0},
        conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"}),
    "nb8b128-pw8-rn16": Recipe(
        "MatMulNBits 8-bit (block 128, symmetric, fp32 compute) incl. pointwise-as-MatMul; ResNet fp16",
        matmul={"mode": "nbits", "bits": 8, "block": 128, "sym": True, "acc": 0},
        conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"}),
    "nb8b128a4-pw8-rn16": Recipe(
        "MatMulNBits 8-bit (block 128, symmetric, accuracy_level=4 → int8 compute) incl. pointwise-as-MatMul; ResNet fp16",
        matmul={"mode": "nbits", "bits": 8, "block": 128, "sym": True, "acc": 4},
        conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"}),
    "nb4-pw4-rn16": Recipe(
        "MatMulNBits 4-bit (block 32, symmetric) incl. pointwise-as-MatMul; ResNet fp16",
        matmul={"mode": "nbits", "bits": 4, "block": 32, "sym": True, "acc": 0},
        conv={"pointwise": "matmul", "resnet": "fp16", "front3d": "fp16"}),
}


# --------------------------------------------------------------------------- graph helpers
def _const_array(node: onnx.NodeProto) -> np.ndarray | None:
    if node.op_type != "Constant":
        return None
    for a in node.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
    return None


def trim_pos_table(model: onnx.ModelProto, max_frames: int) -> dict:
    """Cut the relative-position table to 2*max_frames-1 rows (exact for T <= max_frames)."""
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    inits = {t.name: t for t in g.initializer}

    def table_of(name: str):
        if name in inits:
            return "init", inits[name], np.asarray(numpy_helper.to_array(inits[name]))
        n = producer.get(name)
        if n is not None and n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    return "const", n, numpy_helper.to_array(a.t)
        return None

    def ancestors(name: str, depth: int = 5):
        seen, stack = set(), [(name, 0)]
        while stack:
            cur, d = stack.pop()
            n = producer.get(cur)
            if n is None or n.name in seen or d > depth:
                continue
            seen.add(n.name)
            yield n
            stack.extend((i, d + 1) for i in n.input)

    for sl in (n for n in g.node if n.op_type == "Slice"):
        got = table_of(sl.input[0])
        if got is None:
            continue
        kind, holder, arr = got
        if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] < 1001 or arr.shape[1] % 2 == 0:
            continue
        rows = arr.shape[1]
        center = rows // 2
        if max_frames >= (center + 1):
            return {"trimmed": False, "reason": "max_frames >= table half-width"}
        idx_consts = []
        for inp in sl.input[1:3]:
            hits = [
                n for n in ancestors(inp)
                if n.op_type == "Constant" and (c := _const_array(n)) is not None
                and c.ndim == 0 and int(c) == center
            ]
            if len(hits) != 1:
                raise SystemExit(f"trim_pe: expected exactly one index constant == {center} behind {sl.name}, got {len(hits)}")
            idx_consts.append(hits[0])
        if idx_consts[0].name == idx_consts[1].name:
            raise SystemExit("trim_pe: start/end share one constant (unexpected export layout)")
        for n in idx_consts:
            n.attribute[0].t.CopyFrom(numpy_helper.from_array(np.asarray(max_frames - 1, dtype=np.int64)))
        new = arr[:, center - (max_frames - 1): center + max_frames, :].copy()
        if kind == "init":
            holder.CopyFrom(numpy_helper.from_array(new, holder.name))
        else:
            holder.attribute[0].t.CopyFrom(numpy_helper.from_array(new, holder.attribute[0].t.name))
        return {
            "trimmed": True, "node": sl.name, "rows_before": rows, "rows_after": int(new.shape[1]),
            "saved_mb": round((arr.nbytes - new.nbytes) / 1e6, 1), "max_frames": max_frames,
        }
    return {"trimmed": False, "reason": "no position-table slice found"}


def conv_group(name: str) -> str:
    for grp, pat in CONV_GROUPS.items():
        if re.search(pat, name):
            return grp
    return "other"


def _conv_nodes(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    inits = {t.name for t in model.graph.initializer}
    out: dict[str, list[onnx.NodeProto]] = {}
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) >= 2 and n.input[1] in inits:
            out.setdefault(conv_group(n.name), []).append(n)
    return out


def pointwise_to_matmul(model: onnx.ModelProto) -> int:
    """Conv1d(k=1) on [B, C, T]  ==  Transpose → MatMul([C, Cout]) → (+bias) → Transpose.
    Lets the 24 conformer pointwise convs ride the same per-channel dynamic-int8 MatMul path."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    todo = {n.name for n in _conv_nodes(model).get("pointwise", [])}
    new_nodes: list[onnx.NodeProto] = []
    drop: set[str] = set()
    count = 0
    for n in g.node:
        if n.name not in todo:
            new_nodes.append(n)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        w = inits[n.input[1]]
        if (list(w.dims)[2:] != [1] or attrs.get("group", 1) != 1 or list(attrs.get("strides", [1])) != [1]
                or list(attrs.get("pads", [0, 0])) != [0, 0] or list(attrs.get("dilations", [1])) != [1]):
            new_nodes.append(n)
            continue
        wt = np.ascontiguousarray(numpy_helper.to_array(w)[:, :, 0].T)  # [Cin, Cout]
        wname = f"{w.name}_pwT"
        g.initializer.append(numpy_helper.from_array(wt, wname))
        drop.add(w.name)
        t_in, mm, y = f"{n.name}_pw_in", f"{n.name}_pw_mm", n.output[0]
        new_nodes.append(helper.make_node("Transpose", [n.input[0]], [t_in], name=f"{n.name}_pw_t1", perm=[0, 2, 1]))
        if len(n.input) > 2 and n.input[2]:
            new_nodes.append(helper.make_node("MatMul", [t_in, wname], [mm], name=f"{n.name}_pw_mm"))
            added = f"{n.name}_pw_add"
            new_nodes.append(helper.make_node("Add", [mm, n.input[2]], [added], name=f"{n.name}_pw_bias"))
            new_nodes.append(helper.make_node("Transpose", [added], [y], name=f"{n.name}_pw_t2", perm=[0, 2, 1]))
        else:
            new_nodes.append(helper.make_node("MatMul", [t_in, wname], [mm], name=f"{n.name}_pw_mm"))
            new_nodes.append(helper.make_node("Transpose", [mm], [y], name=f"{n.name}_pw_t2", perm=[0, 2, 1]))
        count += 1
    used = {i for n in new_nodes for i in n.input}
    for t in [t for t in g.initializer if t.name in drop and t.name not in used]:
        g.initializer.remove(t)
    del g.node[:]
    g.node.extend(new_nodes)
    return count


def _select_matmul(model: onnx.ModelProto, exclude: list[str]) -> tuple[list[str], list[str]]:
    """(to quantise, kept fp32) among const-weight MatMuls."""
    inits = {t.name for t in model.graph.initializer}
    pats = [re.compile(p) for p in exclude]
    take, keep = [], []
    for n in model.graph.node:
        if n.op_type != "MatMul" or len(n.input) < 2 or n.input[1] not in inits:
            continue
        (keep if any(p.search(n.name) for p in pats) else take).append(n.name)
    return take, keep


def weight_only_dq8(model: onnx.ModelProto, names: set[str]) -> int:
    """Per-output-channel symmetric int8 weights + DequantizeLinear (fp32 activations)."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    new_nodes: list[onnx.NodeProto] = []
    done: dict[str, str] = {}
    for n in g.node:
        if n.name not in names:
            continue
        w = inits[n.input[1]]
        if w.name in done:
            n.input[1] = done[w.name]
            continue
        arr = numpy_helper.to_array(w).astype(np.float32)
        axis = 0 if n.op_type == "Conv" else arr.ndim - 1  # Conv: Cout; MatMul [K, N]: N
        red = tuple(i for i in range(arr.ndim) if i != axis)
        amax = np.abs(arr).max(axis=red, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
        qn, sn, dn = f"{w.name}_q8", f"{w.name}_q8_scale", f"{w.name}_dq"
        g.initializer.append(numpy_helper.from_array(q, qn))
        g.initializer.append(numpy_helper.from_array(scale.reshape(-1), sn))
        new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn], [dn], name=f"{n.name}_dq8", axis=axis))
        done[w.name] = dn
        n.input[1] = dn
    used = {i for n in list(g.node) + new_nodes for i in n.input}
    for w in [t for t in g.initializer if t.name in done and t.name not in used]:
        g.initializer.remove(w)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)  # DQ only reads initializers → safe at the front
    return len(done)


def weights_fp16(model: onnx.ModelProto, names: set[str]) -> int:
    """Store Conv weights as fp16 + Cast(to=fp32). ORT constant-folds the Cast at load, so compute is fp32
    with prepacked weights; only the file (download) is halved."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    casts: list[onnx.NodeProto] = []
    done: dict[str, str] = {}
    for n in g.node:
        if n.name not in names:
            continue
        w = inits[n.input[1]]
        if w.name not in done:
            arr = numpy_helper.to_array(w).astype(np.float16)
            g.initializer.append(numpy_helper.from_array(arr, f"{w.name}_fp16"))
            casts.append(helper.make_node("Cast", [f"{w.name}_fp16"], [f"{w.name}_f32"],
                                          name=f"{w.name}_cast", to=TensorProto.FLOAT))
            done[w.name] = f"{w.name}_f32"
        n.input[1] = done[w.name]
    used = {i for n in list(g.node) + casts for i in n.input}
    for w in [t for t in g.initializer if t.name in done and t.name not in used]:
        g.initializer.remove(w)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return len(done)


def clear_value_info(model: onnx.ModelProto) -> None:
    del model.graph.value_info[:]


# --------------------------------------------------------------------------- reporting
def _nbytes(t: onnx.TensorProto) -> int:
    return len(t.raw_data) or int(np.prod(t.dims, dtype=np.int64)) * helper.tensor_dtype_to_np_dtype(t.data_type).itemsize


def describe(path: Path) -> dict:
    m = onnx.load(str(path))
    ops = Counter(n.op_type for n in m.graph.node)
    by_type: Counter = Counter()
    for t in m.graph.initializer:
        by_type[TensorProto.DataType.Name(t.data_type)] += _nbytes(t)
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    by_type[TensorProto.DataType.Name(a.t.data_type)] += _nbytes(a.t)
    interesting = {k: v for k, v in ops.items() if k in {
        "MatMul", "MatMulInteger", "MatMulNBits", "Conv", "ConvInteger", "DequantizeLinear", "Cast",
        "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul", "DynamicQuantizeMatMul", "Gemm"}}
    return {
        "file_mb": round(path.stat().st_size / 1e6, 1),
        "weights_mb_by_dtype": {k: round(v / 1e6, 1) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])},
        "ops": interesting,
        "contrib_ops": sorted({n.op_type for n in m.graph.node if n.domain == "com.microsoft"}),
        "opsets": {o.domain or "ai.onnx": o.version for o in m.opset_import},
    }


# --------------------------------------------------------------------------- pipeline
def build(variant: str, src: Path, out: Path, max_frames: int, work: Path, trim: bool = True) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    r = VARIANTS[variant]
    info: dict = {"variant": variant, "desc": r.desc, "src": str(src), "out": str(out)}
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    model = onnx.load(str(src))
    if r.trim_pe and trim:
        info["trim_pe"] = trim_pos_table(model, max_frames)
        print(f"[trim_pe] {info['trim_pe']}")
    if r.conv.get("pointwise") == "matmul":
        info["pointwise_as_matmul"] = pointwise_to_matmul(model)
        print(f"[pointwise→MatMul] {info['pointwise_as_matmul']} convs rewritten")
    cur = work / "step0.surgery.onnx"
    onnx.save(model, str(cur))
    del model

    # Shape-inference preprocessing, no ORT graph optimisation (it would fuse MatMul+Add into Gemm,
    # which the dynamic quantiser does not touch).
    pre = work / "step1.pre.onnx"
    try:
        # auto_merge: the Slice(2T-1) index arithmetic of the position table defeats plain symbolic inference
        quant_pre_process(str(cur), str(pre), skip_optimization=True, auto_merge=True)
        info["preprocess"] = "symbolic+onnx shape inference (auto_merge)"
    except Exception as e:  # noqa: BLE001 - fall back to plain ONNX shape inference
        print(f"[preprocess] symbolic inference failed ({e!r:.80}); using onnx shape inference only")
        quant_pre_process(str(cur), str(pre), skip_optimization=True, skip_symbolic_shape=True)
        info["preprocess"] = "onnx shape inference only"
    cur = pre
    print(f"[preprocess] {info['preprocess']} ({time.perf_counter() - t0:.0f}s)")

    mm = r.matmul
    model = onnx.load(str(cur))
    mm_take, mm_keep = _select_matmul(model, r.exclude)
    groups = _conv_nodes(model)
    info["n_matmul_quantised"], info["n_matmul_fp32_kept"] = (len(mm_take) if mm else 0), (len(mm_keep) if mm else 0)
    info["conv_groups"] = {g: len(v) for g, v in groups.items()}
    del model

    if mm and mm["mode"] == "dyn":
        nxt = work / "step2.mm.onnx"
        quantize_dynamic(
            str(cur), str(nxt), op_types_to_quantize=["MatMul"], per_channel=True,
            weight_type=QuantType.QInt8, nodes_to_exclude=mm_keep,
            extra_options={"MatMulConstBOnly": True},
        )
        cur = nxt
        print(f"[matmul dyn] {len(mm_take)} MatMuls → MatMulInteger ({len(mm_keep)} kept fp32)")
    elif mm and mm["mode"] == "nbits":
        from onnxruntime.quantization import QuantFormat
        from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

        nxt = work / "step2.mm.onnx"
        cfg = DefaultWeightOnlyQuantConfig(
            block_size=mm["block"], is_symmetric=mm["sym"], accuracy_level=mm.get("acc") or None,
            quant_format=QuantFormat.QOperator, op_types_to_quantize=("MatMul",), bits=mm["bits"],
        )
        q = MatMulNBitsQuantizer(onnx.load(str(cur)), bits=mm["bits"], algo_config=cfg, nodes_to_exclude=mm_keep)
        q.process()
        q.model.save_model_to_file(str(nxt), use_external_data_format=False)
        cur = nxt
        print(f"[matmul nbits{mm['bits']}] {len(mm_take)} MatMuls → MatMulNBits")

    dyn_conv = {n.name for g, ns in groups.items() if r.conv.get(g) == "dyn" for n in ns}
    if dyn_conv:
        nxt = work / "step3.conv.onnx"
        other = [n.name for ns in groups.values() for n in ns if n.name not in dyn_conv]
        quantize_dynamic(
            str(cur), str(nxt), op_types_to_quantize=["Conv"], per_channel=False,
            weight_type=QuantType.QUInt8, nodes_to_exclude=other,
        )
        cur = nxt
        print(f"[conv dyn] {len(dyn_conv)} Conv → ConvInteger")

    model = onnx.load(str(cur))
    dq_names = {n.name for g, ns in groups.items() if r.conv.get(g) == "dq8" for n in ns}
    if mm and mm["mode"] == "dq8":
        dq_names |= set(mm_take)
    if dq_names:
        print(f"[dq8] {weight_only_dq8(model, dq_names)} weights → int8 + DequantizeLinear")
    fp16_names = {n.name for g, ns in groups.items() if r.conv.get(g) == "fp16" for n in ns}
    if mm and r.exclude_fp16:
        fp16_names |= set(mm_keep)  # MatMuls excluded from int8: fp16 storage beats fp32 on size
    if fp16_names:
        print(f"[fp16] {weights_fp16(model, fp16_names)} conv weights → fp16 + Cast")
    clear_value_info(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info.update(describe(out))
    return info


# --------------------------------------------------------------------------- sanity check
def quick_check(fp32: Path, cand: Path, exact: bool = False, n_clips: int = 12) -> dict:
    """Cheap self-test: (a) random-input drift vs fp32 at several T (≈0 for the exact surgeries);
    (b) on real LRS3 crops, if the parquet is around: argmax agreement + identical hypotheses."""
    import onnxruntime as ort

    res: dict = {}
    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(cand), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    drift = {}
    for t in (10, 37, 151, 250, 500):
        x = rng.standard_normal((1, 1, t, 88, 88), dtype=np.float32)
        ra, rb = a.run(None, {"video": x})[0], b.run(None, {"video": x})[0]
        drift[t] = float(np.abs(ra - rb).max())
    res["random_max_abs_logp_diff"] = drift
    if exact and max(drift.values()) > 5e-3:  # same tolerance as export_onnx.py's torch-vs-onnx check
        raise SystemExit(f"surgery-only graph deviates from fp32: {drift}")
    pq = ML / "data" / "lrs3_test" / "0000.parquet"
    if pq.is_file():
        sys.path.insert(0, str(ML / "scripts"))
        import bench
        from lipread.model import collapse_ctc, ids_to_text
        from lipread.preprocess import precropped_patches, to_model_input

        tokens = json.loads((ART / "tokens.json").read_text())
        agree, same = [], 0
        clips = bench.clips_from_parquet(pq, n_clips)
        for c in clips:
            x = to_model_input(precropped_patches(c.crops)).unsqueeze(0).numpy()
            ra, rb = a.run(None, {"video": x})[0], b.run(None, {"video": x})[0]
            agree.append(float((ra.argmax(-1) == rb.argmax(-1)).mean()))
            same += ids_to_text(collapse_ctc(ra.argmax(-1).tolist()), tokens) == ids_to_text(collapse_ctc(rb.argmax(-1).tolist()), tokens)
        res["lrs3_argmax_agree_mean"] = float(np.mean(agree))
        res["lrs3_identical_hyp"] = f"{same}/{len(clips)}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=None, help="default artifacts/lipread_ctc.<variant>.onnx")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="position-table capacity (frames)")
    ap.add_argument("--no-trim-pe", action="store_true", help="keep the 30.7 MB position table")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()

    if a.list or not a.variant:
        for k, v in VARIANTS.items():
            print(f"{k:16s} {v.desc}")
        return
    out = a.out or ART / f"lipread_ctc.{a.variant}.onnx"
    if out.resolve() == a.src.resolve():
        raise SystemExit("refusing to overwrite the source model")
    work = ART / "_work" / a.variant
    info = build(a.variant, a.src, out, a.max_frames, work, trim=not a.no_trim_pe)
    if not a.no_check:
        r = VARIANTS[a.variant]
        info["check"] = quick_check(a.src, out, exact=(r.matmul is None and not any(
            m in ("dyn", "dq8", "fp16") for m in r.conv.values())))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(info, indent=1))
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
