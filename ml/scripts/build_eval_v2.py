"""Build raw_eval_v2: ~150 short natural-sentence clips of faces the model never trained on, with
published-script transcripts that Whisper confirms on each clip's audio, so WER can be broken down by
speaker, sex, age, skin tone, lighting, pose, source.

Why: raw_eval is 20 clips / 122 words (runs swing 2-4 points) and has few faces; LRS3-test crops skip
our crop path (brief D33). The checkpoint (Auto-AVSR LRS3_V_WER19.1) trained on LRS2, LRS3, VoxCeleb2
and AVSpeech, so none of those. GRID is left out (command grammar, not how people talk), and so are
the speakers already in raw_eval: the app's thresholds were tuned on them.

    cd ml
    uv run python scripts/build_eval_v2.py                      # every source we can reach
    uv run python scripts/build_eval_v2.py --sources cremad --dry-run
    uv run python scripts/build_eval_v2.py --measure-only       # recompute measured attributes
    uv run python scripts/build_eval_v2.py --regroup            # after editing annotations.json

Output (data/raw_eval_v2, gitignored; the private HF dataset holds the copy):
  <id>.mp4 + <id>.txt         open-licence sources, flat (`bench.py --clips data/raw_eval_v2`)
  <source>/<id>.mp4 + .txt    each gated or unclear-terms source (MEAD), own folder + TERMS.txt
  manifest.json               per clip: transcript, source, licence, speaker attributes, measured
                              face size / pose / motion / lighting / skin tone, ASR check
  annotations.json            hand labels per speaker (apparent skin tone, coarse Monk bands)
  ATTRIBUTION.txt
Clips are H.264 mp4, CFR at the source fps, no audio (like raw_eval). Picks are seeded and balanced
(each sentence spread over the groups), so a rebuild gives the same set. Downloads are cached in
data/v2_cache; zip archives are read by HTTP range, one member at a time; MEAD tars are streamed once.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import numpy as np

ML = Path(__file__).resolve().parents[1]
OUT = ML / "data" / "raw_eval_v2"
CACHE = ML / "data" / "v2_cache"
V1_SOURCES = ML / "data" / "raw_eval" / "sources.json"
TRAINED_ON = "LRS2, LRS3, VoxCeleb2, AVSpeech (Auto-AVSR, Ma et al. 2023: 3,448 h)"

# Inter-ocular distance of the mean face the crop warps to (20words_mean_face.npy, 256 px frame).
# The 96 px mouth crop spans 96 / IOD_REF = 1.77 eye-distances, so a face with iod_px < IOD_REF is
# upsampled into the crop (blur), and crop_scale = IOD_REF / iod_px says by how much.
IOD_REF = 54.3


@dataclass
class Clip:
    id: str
    source: str               # key in SOURCES
    speaker: str              # globally unique, e.g. cremad_1045
    transcript: str
    url: str                  # where the bytes come from (archive URL :: member for zips)
    fetch: Callable[[Path], Path] = field(repr=False, compare=False)  # → local source file
    sex: str = "unknown"      # F / M, as the dataset labels it
    age: int | None = None
    race: str = "not labelled"       # dataset's own label, verbatim
    ethnicity: str = "not labelled"
    extra: dict = field(default_factory=dict)  # emotion, view, sentence code, ...
    trim: tuple[float, float] | None = None    # seconds, when the clip is a span of a longer file


@dataclass
class Source:
    key: str
    name: str
    licence: str
    cite: str
    home: str
    gated: bool               # True → own subfolder + TERMS.txt (gated, or terms that need tracing)
    candidates: Callable[[], list[Clip]]
    pick: Callable[[list[Clip], random.Random], list[Clip]]
    terms_note: str = ""
    access: str = "open"      # how we got it: open / HF terms accepted / web licence agreement / ...


# ── downloads ────────────────────────────────────────────────────────────────────────────────────

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0))
    return _client


def download(url: str, dst: Path) -> Path:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with client().stream("GET", url) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dst)
    return dst


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests: lets zipfile pull one member of a big
    archive (RAVDESS actor zips are ~550 MB; one clip is ~5 MB)."""

    def __init__(self, url: str):
        self.url = url
        r = client().head(url)
        r.raise_for_status()
        self.size = int(r.headers["content-length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = client().get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        data = r.content
        if r.status_code == 200:  # server ignored the range
            data = data[self.pos: end + 1]
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


_zips: dict[str, zipfile.ZipFile] = {}


def zip_member(url: str, member: str, dst: Path) -> Path:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    if url not in _zips:
        _zips[url] = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=4 << 20))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with _zips[url].open(member) as src, open(tmp, "wb") as f:
        shutil.copyfileobj(src, f, 4 << 20)
    tmp.rename(dst)
    return dst


def v1_excluded() -> set[str]:
    """Speakers already in raw_eval (v1): keep v2 disjoint from what the app was tuned on."""
    if not V1_SOURCES.is_file():
        return set()
    out = set()
    for cid, s in json.loads(V1_SOURCES.read_text()).items():
        m = re.match(r"actor (\d+)", s.get("speaker", ""))
        if m and s["dataset"] == "CREMA-D":
            out.add(f"cremad_{m.group(1)}")
        elif m and s["dataset"] == "RAVDESS":
            out.add(f"ravdess_a{int(m.group(1)):02d}")
    return out


def spread(clips: list[Clip], n_per_speaker: int, rng: random.Random, item_key: str) -> list[Clip]:
    """Take n clips per speaker, rotating through items (sentences) so every item lands about
    equally often in every group the speakers were ordered by."""
    by_spk: dict[str, list[Clip]] = defaultdict(list)
    for c in clips:
        by_spk[c.speaker].append(c)
    items = sorted({c.extra[item_key] for c in clips})
    rng.shuffle(items)
    out, k = [], 0
    for spk in by_spk:  # caller orders speakers (stratified)
        have = {c.extra[item_key]: c for c in by_spk[spk]}
        took = 0
        for _ in range(len(items)):
            it = items[k % len(items)]
            k += 1
            if it in have:
                out.append(have[it])
                took += 1
                if took == n_per_speaker:
                    break
    return out


def stratified_speakers(meta: dict[str, dict], keys: tuple[str, ...], n: int, rng: random.Random) -> list[str]:
    """Pick n speakers round-robin over strata (e.g. race × sex × age band), so small groups are
    taken whole before large ones fill up. Returns them interleaved by stratum."""
    strata: dict[tuple, list[str]] = defaultdict(list)
    for spk, m in sorted(meta.items()):
        strata[tuple(m[k] for k in keys)].append(spk)
    for v in strata.values():
        rng.shuffle(v)
    order = sorted(strata, key=lambda s: (len(strata[s]), s))
    picked: list[str] = []
    while len(picked) < n and any(strata[s] for s in order):
        for s in order:
            if len(picked) == n:
                break
            if strata[s]:
                picked.append(strata[s].pop())
    return picked


def age_band(age: int | None) -> str:
    if age is None:
        return "unknown"
    return "18-29" if age < 30 else "30-44" if age < 45 else "45-59" if age < 60 else "60+"


# ── sources ──────────────────────────────────────────────────────────────────────────────────────

# CREMA-D: 91 actors (ages 20-74; African American, Asian, Caucasian; Hispanic or not), 12 fixed
# sentences, 6 emotions; 480x360 FLV at 30 fps. Demographics per actor in VideoDemographics.csv.
CREMAD_RAW = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master"
CREMAD_MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master"
CREMAD_SENTENCES = {
    "IEO": "It's eleven o'clock", "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting", "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before",
    "TSI": "The surface is slick", "WSI": "We'll stop in a couple of minutes",
}
CREMAD_BAD = {"1076_MTI_NEU_XX", "1076_MTI_SAD_XX", "1064_TIE_SAD_XX", "1064_IEO_DIS_MD"}  # README
CREMAD_SPEAKERS = 30
CREMAD_PER_SPEAKER = 2


def cremad_meta() -> dict[str, dict]:
    f = download(f"{CREMAD_RAW}/VideoDemographics.csv", CACHE / "cremad" / "VideoDemographics.csv")
    out = {}
    for r in csv.DictReader(open(f)):
        sex = {"Male": "M", "Female": "F"}.get(r["Sex"], "unknown")
        out[f"cremad_{r['ActorID']}"] = {
            "actor": r["ActorID"], "age": int(r["Age"]), "sex": sex, "race": r["Race"],
            "ethnicity": r["Ethnicity"], "age_band": age_band(int(r["Age"])),
        }
    return out


def cremad_candidates() -> list[Clip]:
    meta = cremad_meta()
    names = download(f"{CREMAD_RAW}/SentenceFilenames.csv", CACHE / "cremad" / "SentenceFilenames.csv")
    out = []
    for r in csv.DictReader(open(names)):
        name = r["Filename"]
        actor, sent, emo, level = name.split("_")
        if emo != "NEU" or name in CREMAD_BAD:  # neutral: closest to someone talking normally
            continue
        spk = f"cremad_{actor}"
        m = meta[spk]
        url = f"{CREMAD_MEDIA}/VideoFlash/{name}.flv"
        out.append(Clip(
            id=f"cremad_{actor}_{sent.lower()}", source="cremad", speaker=spk,
            transcript=CREMAD_SENTENCES[sent], url=url,
            fetch=lambda d, url=url, name=name: download(url, d / f"{name}.flv"),
            sex=m["sex"], age=m["age"], race=m["race"], ethnicity=m["ethnicity"],
            extra={"sentence": sent, "emotion": "neutral"},
        ))
    return out


def cremad_pick(cands: list[Clip], rng: random.Random) -> list[Clip]:
    meta = {k: v for k, v in cremad_meta().items() if k not in v1_excluded()}
    # Every Asian and "Unknown" actor, then African American and Caucasian evenly by sex × age.
    groups = {k: {**v, "grp": v["race"] if v["race"] in ("African American", "Caucasian") else "other"}
              for k, v in meta.items()}
    spk = stratified_speakers(groups, ("grp", "sex", "age_band"), CREMAD_SPEAKERS, rng)
    order = {s: i for i, s in enumerate(spk)}
    chosen = sorted((c for c in cands if c.speaker in order), key=lambda c: (order[c.speaker], c.id))
    return spread(chosen, CREMAD_PER_SPEAKER, rng, "sentence")


# RAVDESS: 24 actors (12 F, 12 M), 2 sentences, 1280x720 mp4 at 29.97 fps, per-actor zips on Zenodo.
# Filename: modality-vocal-emotion-intensity-statement-repetition-actor (odd actor = male).
RAVDESS_ZIP = "https://zenodo.org/api/records/1188976/files/Video_Speech_Actor_{:02d}.zip/content"
RAVDESS_SENTENCES = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}
RAVDESS_SPEAKERS = 8


def ravdess_candidates() -> list[Clip]:
    out = []
    for actor in range(1, 25):
        url = RAVDESS_ZIP.format(actor)
        for stmt in (1, 2):
            # full AV (01) so the ASR check can hear it; neutral (01), normal intensity, repetition 1
            member = f"Actor_{actor:02d}/01-01-01-01-{stmt:02d}-01-{actor:02d}.mp4"
            out.append(Clip(
                id=f"ravdess_a{actor:02d}_s{stmt}", source="ravdess", speaker=f"ravdess_a{actor:02d}",
                transcript=RAVDESS_SENTENCES[stmt], url=f"{url} :: {member}",
                fetch=lambda d, url=url, member=member: zip_member(url, member, d / Path(member).name),
                sex="M" if actor % 2 else "F", extra={"sentence": f"s{stmt}", "emotion": "neutral"},
            ))
    return out


def ravdess_pick(cands: list[Clip], rng: random.Random) -> list[Clip]:
    excl = v1_excluded()
    actors = sorted({c.speaker for c in cands} - excl)
    f = [a for a in actors if int(a[-2:]) % 2 == 0]
    m = [a for a in actors if int(a[-2:]) % 2 == 1]
    rng.shuffle(f)
    rng.shuffle(m)
    # f, m, m, f, f, m, ...: with one clip each and two sentences rotating, both sexes get both
    spk = [x for i, (a, b) in enumerate(zip(f, m)) for x in ((a, b) if i % 2 == 0 else (b, a))]
    spk = spk[:RAVDESS_SPEAKERS]
    order = {s: i for i, s in enumerate(spk)}
    chosen = sorted((c for c in cands if c.speaker in order), key=lambda c: (order[c.speaker], c.id))
    return spread(chosen, 1, rng, "sentence")


# VidTIMIT: 43 people (19 F / 24 M; sex = id prefix), 10 TIMIT sentences each as 512x384 JPEG frames
# at 25 fps + 32 kHz wav, office lighting, per-speaker zips on Zenodo. Only the two TIMIT dialect
# sentences every speaker reads (sa1, sa2) have a published text; the other 8 prompts are LDC-licensed
# and not shipped, and Whisper agreement is not exact enough to stand in for them (it got 3 of 8
# MEAD takes wrong), so they are not used.
VIDTIMIT_ZIP = "https://zenodo.org/api/records/158963/files/{}.zip/content"
VIDTIMIT_IDS = ("fadg0 faks0 fcft0 fcmh0 fcmr0 fcrh0 fdac1 fdms0 fdrd1 fedw0 felc0 fgjd0 fjas0 fjem0 "
                "fjre0 fjwb0 fkms0 fpkt0 fram1 mabw0 mbdg0 mbjk0 mccs0 mcem0 mdab0 mdbb0 mdld0 mgwt0 "
                "mjar0 mjsw0 mmdb1 mmdm2 mpdf0 mpgl0 mrcz0 mreb0 mrgg0 mrjo0 msjs1 mstk0 mtas1 mtmr0 "
                "mwbt0").split()
VIDTIMIT_SA = {"sa1": "She had your dark suit in greasy wash water all year",
               "sa2": "Don't ask me to carry an oily rag like that"}
VIDTIMIT_SPEAKERS = 16


def vidtimit_fetch(spk: str, sent: str, d: Path) -> Path:
    """Frames + wav of one sentence (range reads from the speaker zip) → lossless mkv with audio."""
    dst = d / f"{spk}_{sent}.mkv"
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    url = VIDTIMIT_ZIP.format(spk)
    if url not in _zips:
        _zips[url] = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=4 << 20))
    z = _zips[url]
    frames = sorted(n for n in z.namelist() if n.startswith(f"{spk}/video/{sent}/") and not n.endswith("/"))
    if not frames:
        raise FileNotFoundError(f"{spk}/{sent}: no frames")
    with tempfile.TemporaryDirectory() as t:
        for i, n in enumerate(frames, 1):
            (Path(t) / f"{i:04d}.jpg").write_bytes(z.read(n))
        (Path(t) / "a.wav").write_bytes(z.read(f"{spk}/audio/{sent}.wav"))
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "25", "-i", f"{t}/%04d.jpg",
                        "-i", f"{t}/a.wav", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
                        "-c:a", "flac", "-shortest", str(dst)], check=True)
    return dst


def vidtimit_candidates() -> list[Clip]:
    return [Clip(
        id=f"vidtimit_{spk}_{sent}", source="vidtimit", speaker=f"vidtimit_{spk}",
        transcript=text, url=f"{VIDTIMIT_ZIP.format(spk)} :: {spk}/video/{sent}/",
        fetch=lambda d, spk=spk, sent=sent: vidtimit_fetch(spk, sent, d),
        sex="F" if spk[0] == "f" else "M", extra={"sentence": sent})
        for spk in VIDTIMIT_IDS for sent, text in VIDTIMIT_SA.items()]


def vidtimit_pick(cands: list[Clip], rng: random.Random) -> list[Clip]:
    """16 speakers, 8 F + 8 M, both sentences each (a clip that fails the transcript check is
    dropped; its speaker keeps the other sentence)."""
    f = sorted({c.speaker for c in cands if c.sex == "F"})
    m = sorted({c.speaker for c in cands if c.sex == "M"})
    rng.shuffle(f)
    rng.shuffle(m)
    spk = [x for pair in zip(f, m) for x in pair][:VIDTIMIT_SPEAKERS]
    return [c for s_ in spk for c in cands if c.speaker == s_]


# MEAD: 1920x1080 30 fps, 7 cameras filming the same take (front, left/right 30 and 60, top, down),
# so one sentence from several angles isolates head pose. Speech clips carry audio; the transcript is
# what two Whisper models agree on (the per-clip sentence list is only in the paper's supplement).
# Bytes come from an unofficial HF copy of the public Google Drive release (per-actor tars, members in
# random order, so each tar is streamed once and only the neutral clips are kept).
MEAD_HF = "https://huggingface.co/datasets/jordantencent/my_MEAD/resolve/main/{}"
MEAD_ACTORS = {"W024": ("W024_video.tar", "F"), "W028": ("W028_video.tar", "F"),
               "W025": ("W025_video.tar", "F"), "W029": ("W029_video.tar", "F"),
               "M009": ("video_m9.tar", "M"), "M011": ("video_m11.tar", "M"),
               "M005": ("video_m5.tar", "M"), "M012": ("video_m12.tar", "M")}
MEAD_SUPP = "https://wywu.github.io/projects/MEAD/support/MEAD-supp.pdf"  # sentence list, section 6
MEAD_SENTENCES_PER_ACTOR = 2
# each sentence: front + a 30-degree side + one of top / down / 60-degree side
MEAD_VIEW_SETS = [("front", "left_30", "down"), ("front", "right_30", "top"),
                  ("front", "right_30", "left_60"), ("front", "left_30", "right_60")]


def mead_extract(actor: str) -> Path:
    """Stream one actor tar (members are in random order, so no range shortcut) and keep only the
    neutral clips: ~90 s per 4-6 GB tar at ~45 MB/s, ~300 MB kept. Cached; resumes from scratch."""
    import tarfile
    d = CACHE / "mead" / actor
    done = d / "COMPLETE"
    if done.is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    url = MEAD_HF.format(MEAD_ACTORS[actor][0])
    with client().stream("GET", url) as r:
        r.raise_for_status()
        raw = r.iter_raw(1 << 20)

        class Stream(io.RawIOBase):
            buf = b""

            def readable(self):
                return True

            def readinto(self, b):
                while not self.buf:
                    try:
                        self.buf = next(raw)
                    except StopIteration:
                        return 0
                n = min(len(b), len(self.buf))
                b[:n], self.buf = self.buf[:n], self.buf[n:]
                return n

        with tarfile.open(fileobj=io.BufferedReader(Stream(), 1 << 20), mode="r|") as t:
            for m in t:
                mm = re.fullmatch(r"video/(\w+)/neutral/level_1/(\d+)\.mp4", m.name)
                if mm and m.isfile():
                    (d / f"{mm.group(1)}_{mm.group(2)}.mp4").write_bytes(t.extractfile(m).read())
    done.write_text("ok\n")
    return d


def mead_sentences() -> list[str]:
    """The ~140 sentences MEAD actors read, from the paper's supplementary PDF (fetched, not vendored)."""
    txt = CACHE / "mead" / "MEAD-supp.txt"
    if not txt.is_file():
        pdf = download(MEAD_SUPP, CACHE / "mead" / "MEAD-supp.pdf")
        try:
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
        except FileNotFoundError:
            import pypdf  # fallback when poppler isn't installed
            txt.write_text("\n".join(pg.extract_text() for pg in pypdf.PdfReader(str(pdf)).pages))
    found = re.finditer(r"^\s*\d+\.\s+([A-Z][^\n]{8,})$", txt.read_text(), re.M)
    return sorted({re.sub(r"\s+", " ", m.group(1)).strip().replace("’", "'") for m in found})


def mead_candidates() -> list[Clip]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        dirs = dict(zip(MEAD_ACTORS, ex.map(mead_extract, MEAD_ACTORS)))
    out = []
    for actor, (tar, sex) in MEAD_ACTORS.items():
        for f in sorted(dirs[actor].glob("*.mp4")):
            view, n = f.stem.rsplit("_", 1)
            name = f"video/{view}/neutral/level_1/{n}.mp4"
            out.append(Clip(
                id=f"mead_{actor.lower()}_{n}_{view}", source="mead", speaker=f"mead_{actor.lower()}",
                transcript="", url=f"{MEAD_HF.format(tar)} :: {name}", fetch=lambda d, f=f: f,
                sex=sex, extra={"sentence": n, "view": view, "emotion": "neutral"}))
    return out


def mead_pick(cands: list[Clip], rng: random.Random) -> list[Clip]:
    """Per actor: takes whose views are all there; the script is the list sentence closest to what
    Whisper hears on the front camera, kept only if judge() confirms it; then that take from 3 angles."""
    import jiwer
    sentences = mead_sentences()
    out = []
    for i, actor in enumerate(sorted({c.speaker for c in cands})):
        mine = [c for c in cands if c.speaker == actor]
        views = {(c.extra["sentence"], c.extra["view"]): c for c in mine}
        takes = sorted({c.extra["sentence"] for c in mine})
        rng.shuffle(takes)
        took = 0
        for take in takes:
            vs = MEAD_VIEW_SETS[(2 * i + took) % len(MEAD_VIEW_SETS)]
            if not all((take, v) in views for v in vs):
                continue
            front = views[(take, "front")]
            hyps = transcribe(front.fetch(CACHE / "mead"), None, ASR_MODELS)
            if not hyps:
                continue
            heard = list(hyps.values())[-1]
            script = min(sentences, key=lambda t: jiwer.wer(_canon(t), _canon(heard) or "x"))
            chk = judge(script, hyps)
            if chk["status"] not in ("match", "near_match"):
                print(f"  mead {actor} take {take}: {chk['status']} vs list sentence {script!r}: {hyps}")
                continue
            for v in vs:
                views[(take, v)].transcript = script
                views[(take, v)].extra["take_check"] = {**chk, "script_from": "MEAD supplementary sentence list"}
            out += [views[(take, v)] for v in vs]
            took += 1
            if took == MEAD_SENTENCES_PER_ACTOR:
                break
    return out


SOURCES: dict[str, Source] = {
    "cremad": Source(
        "cremad", "CREMA-D", "ODbL 1.0 (database) + DbCL 1.0 (contents)",
        "Cao et al. (2014), CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset, "
        "IEEE Trans. Affective Computing 5(4)", "https://github.com/CheyneyComputerScience/CREMA-D",
        False, cremad_candidates, cremad_pick),
    "ravdess": Source(
        "ravdess", "RAVDESS", "CC BY-NC-SA 4.0",
        "Livingstone & Russo (2018), The Ryerson Audio-Visual Database of Emotional Speech and Song "
        "(RAVDESS), PLoS ONE 13(5): e0196391", "https://zenodo.org/records/1188976",
        False, ravdess_candidates, ravdess_pick),
    "vidtimit": Source(
        "vidtimit", "VidTIMIT", "CC BY-NC-ND 4.0 (Zenodo); cite Sanderson & Lovell (2009)",
        "C. Sanderson and B.C. Lovell, Multi-Region Probabilistic Histograms for Robust and Scalable "
        "Identity Inference, LNCS 5558, 2009", "https://zenodo.org/records/158963",
        False, vidtimit_candidates, vidtimit_pick),
    "mead": Source(
        "mead", "MEAD", "data licence not stated (code MIT); research use only",
        "Wang et al. (2020), MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face "
        "Generation, ECCV", "https://wywu.github.io/projects/MEAD/MEAD.html",
        True, mead_candidates, mead_pick,
        terms_note="no data licence published; used for private research evaluation only; bytes from "
                   "the unofficial HF copy jordantencent/my_MEAD of the public Google Drive release",
        access="open (public Google Drive release; data licence not stated)"),
}


# ── normalise ────────────────────────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = (int(x) for x in (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/"))
    fps = num / den if den else 0.0
    if not 1 <= fps <= 120:
        num, den = (int(x) for x in s["r_frame_rate"].split("/"))
        fps = num / den
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": round(fps, 3),
            "seconds": round(float(j["format"].get("duration", 0.0)), 3)}


def normalise(src: Path, dst: Path, trim: tuple[float, float] | None) -> dict:
    """→ H.264 mp4, constant frame rate at the source rate, no audio (what raw_eval holds)."""
    fps = probe(src)["fps"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    cut = ["-ss", f"{trim[0]:.3f}", "-to", f"{trim[1]:.3f}"] if trim else []
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", *cut, "-i", str(src), "-map", "0:v:0", "-an",
         "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    return probe(dst)


def extract_audio(src: Path, trim: tuple[float, float] | None) -> Path | None:
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    cut = ["-ss", f"{trim[0]:.3f}", "-to", f"{trim[1]:.3f}"] if trim else []
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *cut, "-i", str(src), "-vn", "-ac", "1",
                        "-ar", "16000", str(tmp)], capture_output=True)
    if r.returncode or tmp.stat().st_size < 2000:
        tmp.unlink(missing_ok=True)
        return None
    return tmp


# ── measure ──────────────────────────────────────────────────────────────────────────────────────
# Everything is measured on the 25 fps frames the model path sees, with the same BlazeFace detector
# (MouthCropper.landmarks: short range, full range as fallback). Keypoints: right eye, left eye,
# nose tip, mouth centre, in source pixels.

def _ita(rgb_patches: list[np.ndarray]) -> float | None:
    """Individual Typology Angle (Chardon 1991) of skin pixels: atan2(L* - 50, b*) in degrees.
    Higher = lighter. Sensitive to white balance and light, so it is binned coarsely."""
    from skimage.color import rgb2lab
    px = np.concatenate([p.reshape(-1, 3) for p in rgb_patches if p.size], axis=0) if rgb_patches else None
    if px is None or len(px) < 50:
        return None
    lab = rgb2lab((px.astype(np.float32) / 255.0)[None])[0]
    L, b = lab[:, 0], lab[:, 2]
    keep = (L > 15) & (L < 95)  # drop specular highlights and deep shadow
    if keep.sum() < 50:
        return None
    return float(math.degrees(math.atan2(np.median(L[keep]) - 50.0, np.median(b[keep]))))


def measure(path: Path, cropper) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """→ (measured attributes, a mid-clip 96x96 mouth crop, a face thumbnail) for one clip."""
    import cv2

    from lipread.preprocess import NoFaceError
    from lipread.video import load_video_25fps
    frames = load_video_25fps(path)
    try:
        lms = cropper.landmarks(frames)
    except NoFaceError as e:
        return {"face_cov": 0.0, "error": str(e)}, None, None
    det = [(i, np.asarray(lm, dtype=np.float32)) for i, lm in enumerate(lms) if lm is not None]
    cov = len(det) / len(frames)
    re_, le, nose, mouth = (np.stack([lm[k] for _, lm in det]) for k in range(4))
    eyes = (re_ + le) / 2
    iod = np.linalg.norm(le - re_, axis=1)
    iod_med = float(np.median(iod))
    yaw = (nose[:, 0] - eyes[:, 0]) / iod
    roll = np.degrees(np.arctan2(le[:, 1] - re_[:, 1], le[:, 0] - re_[:, 0]))
    # pitch proxy: nose→mouth over eyes→nose vertical spans (looking down shrinks the upper span)
    pitch = (mouth[:, 1] - nose[:, 1]) / np.maximum(nose[:, 1] - eyes[:, 1], 1e-3)
    centre = eyes / iod_med
    motion = float(np.sqrt(centre[:, 0].var() + centre[:, 1].var()))
    step = np.linalg.norm(np.diff(centre, axis=0), axis=1) if len(centre) > 1 else np.zeros(1)

    lum, hi, eye_hi, asym, frame_lum, patches = [], [], [], [], [], []
    for i, lm in det[:: max(1, len(det) // 12)]:
        f = frames[i]
        h, w = f.shape[:2]
        y = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        frame_lum.append(float(np.median(y)))
        rt, lf, ns, mo = lm
        d = float(np.linalg.norm(lf - rt))
        ex, ey = (rt + lf) / 2
        x0, x1 = int(max(0, ex - 1.1 * d)), int(min(w, ex + 1.1 * d))
        y0, y1 = int(max(0, ey - 0.7 * d)), int(min(h, mo[1] + 0.6 * d))
        face = y[y0:y1, x0:x1]
        if face.size:
            lum.append(float(face.mean()))
            hi.append(float(np.percentile(face, 95)))  # highlights: dim light, not dark skin
            nx = int(np.clip(ns[0] - x0, 1, face.shape[1] - 1))
            a, b_ = float(face[:, :nx].mean()), float(face[:, nx:].mean())
            asym.append(abs(a - b_) / max(a + b_, 1e-3))
        q = max(2, int(0.2 * d))  # eye whites + catchlights: bright under good light whatever the skin
        eyes_px = np.concatenate([y[max(0, int(e[1]) - q): int(e[1]) + q, max(0, int(e[0]) - q): int(e[0]) + q].ravel()
                                  for e in (rt, lf)])
        if eyes_px.size:
            eye_hi.append(float(np.percentile(eyes_px, 99)))
        s = max(2, int(0.15 * d))
        for e in (rt, lf):  # cheek just below each eye
            cx, cy = int(e[0]), int(e[1] + 0.65 * d)
            patches.append(f[max(0, cy - s): cy + s, max(0, cx - s): cx + s])

    crop = thumb = None
    crop_stats = {}
    try:
        crops = cropper.crop(frames)
        crop = crops[len(crops) // 2]
        x = crops[:, 4:92, 4:92].astype(np.float32)  # the 88x88 the model sees
        mouth = x[:, 24:64, 14:74]                    # crops are centred on the mouth
        crop_stats = {
            "crop_contrast": round(float(np.mean([f.std() for f in x])), 2),       # RMS, gray levels
            "mouth_luma": round(float(mouth.mean()), 1),
            # articulation: mean frame-to-frame change around the mouth (gray levels / frame)
            "articulation": round(float(np.abs(np.diff(mouth, axis=0)).mean()), 3) if len(mouth) > 1 else 0.0,
        }
    except NoFaceError:
        pass
    i, lm = det[len(det) // 2]
    rt, lf, ns, mo = lm
    d = float(np.linalg.norm(lf - rt))
    ex, ey = (rt + lf) / 2
    h, w = frames.shape[1:3]
    x0, x1 = int(max(0, ex - 1.4 * d)), int(min(w, ex + 1.4 * d))
    y0, y1 = int(max(0, ey - 1.1 * d)), int(min(h, mo[1] + 1.0 * d))
    if x1 > x0 and y1 > y0:
        thumb = cv2.resize(frames[i][y0:y1, x0:x1], (160, int(160 * (y1 - y0) / (x1 - x0))))

    meas = {
        "face_cov": round(cov, 3), "frames_25fps": int(len(frames)),
        "iod_px": round(iod_med, 1), "crop_scale": round(IOD_REF / iod_med, 2),
        "yaw": round(float(np.median(yaw)), 3), "yaw_range": round(float(np.ptp(yaw)), 3),
        "roll_deg": round(float(np.median(roll)), 1), "pitch_ratio": round(float(np.median(pitch)), 2),
        "motion": round(motion, 4), "motion_step_p95": round(float(np.percentile(step, 95)), 4),
        "face_luma": round(float(np.median(lum)), 1) if lum else None,
        "face_p95": round(float(np.median(hi)), 1) if hi else None,
        "eye_p99": round(float(np.median(eye_hi)), 1) if eye_hi else None,
        "frame_luma": round(float(np.median(frame_lum)), 1) if frame_lum else None,
        "side_light": round(float(np.median(asym)), 3) if asym else None,
        "ita": None if (ita := _ita(patches)) is None else round(ita, 1),
        **crop_stats,
    }
    return meas, crop, thumb


def groups(c: dict, skin_labels: dict[str, str] | None = None) -> dict:
    """Bins for the breakdown. Thresholds are coarse on purpose (see the report for the spread).
    skin_tone comes from annotations.json (by-eye Monk bands, or a dataset's own skin-type label);
    the measured ITA bin is kept as skin_ita because video white balance shifts it a lot."""
    m = c["measured"]
    ita = m.get("ita")
    skin_ita = ("unknown" if ita is None else "light" if ita > 41 else "intermediate" if ita > 28
                else "tan" if ita > 10 else "dark")
    skin = (skin_labels or {}).get(c["speaker"], "unlabelled")
    # pose: the camera angle where the dataset gives it (MEAD), else measured yaw = nose offset from
    # the eye midpoint in eye-distances (MEAD's 30-degree cameras measure 0.38-0.62, 60-degree ~1.0)
    yaw = abs(m.get("yaw") or 0.0)
    view = c.get("view")
    pose = ({"front": "frontal", "top": "camera above", "down": "camera below"}.get(view)
            or ("turned 30°" if view and view.endswith("30") else "turned 60°" if view else None)
            or ("frontal" if yaw < 0.2 else "turned 30°" if yaw < 0.75 else "turned 60°"))
    # side-lit = one half of the face much darker than the other. No "dim" class: none of the open
    # sources is dim-lit, and every brightness cue tried (face p95, eye-region p99) flagged darker
    # skin under studio light instead (kept in "measured" for reference).
    light = "side-lit" if (m.get("side_light") or 0) > 0.12 else "even"
    cs = m.get("crop_scale") or 1.0
    crop = "upsampled" if cs > 1.25 else "downsampled" if cs < 0.8 else "native"
    motion = "moving" if (m.get("motion") or 0) > 0.05 else "still"
    return {"sex": c["sex"], "age_band": age_band(c.get("age")), "skin_tone": skin, "skin_ita": skin_ita,
            "lighting": light,
            "pose": pose, "crop": crop, "motion": motion, "source": c["source_name"]}


# ── ASR check ────────────────────────────────────────────────────────────────────────────────────

def norm_words(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_asr: dict[str, object] = {}
ASR_MODELS: list[str] = ["small.en", "medium.en", "large-v3-turbo"]  # --asr sets this


def asr(wav: Path, model: str) -> str:
    from faster_whisper import WhisperModel
    if model not in _asr:
        _asr[model] = WhisperModel(model, device="cpu", compute_type="int8")
    segs, _ = _asr[model].transcribe(str(wav), language="en", beam_size=5, vad_filter=False,
                                     condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip()


_US = {w + "our": w + "or" for w in ("hon", "col", "fav", "lab", "neighb", "behavi", "flav", "hum",
                                      "rum", "harb", "vap", "arm", "od", "endeav", "vig", "splend")}
_US.update({"centre": "center", "theatre": "theater", "metre": "meter", "grey": "gray",
            "programme": "program", "travelled": "traveled", "travelling": "traveling",
            "jewellery": "jewelry", "cheque": "check", "tyre": "tire", "aluminium": "aluminum",
            "realise": "realize", "realised": "realized", "organise": "organize",
            "organised": "organized", "recognise": "recognize", "recognised": "recognized",
            "apologise": "apologize", "analyse": "analyze", "catalogue": "catalog"})


def american(text: str) -> str:
    """British → American spelling, keeping case and punctuation (the VSR writes American)."""
    def fix(m):
        w = m.group(0)
        u = _US.get(w.lower())
        if u is None and w.lower().endswith("s") and w.lower()[:-1] in _US:
            u = _US[w.lower()[:-1]] + "s"
        return w if u is None else (u.capitalize() if w[0].isupper() else u)
    return re.sub(r"[A-Za-z]+", fix, text)


def asr_equal(a: str, b: str) -> bool:
    """Same words, allowing spelled-out numbers / contractions Whisper writes differently."""
    canon = {"o'clock": "oclock", "eleven": "11", "doctor's": "doctors", "i'm": "i am",
             "i've": "i have", "we'll": "we will", "don't": "do not", "it's": "it is",
             "that's": "that is", "can't": "cannot", "won't": "will not", "you're": "you are"}
    def c(s):
        s = norm_words(american(s)).replace("o'clock", "oclock")
        return " ".join(canon.get(w, w) for w in s.split()).replace("11 oclock", "eleven oclock")
    return c(a) == c(b)


def transcribe(src: Path, trim: tuple[float, float] | None, models: list[str],
               script: str = "") -> dict[str, str] | None:
    """Whisper hypotheses for the clip's own audio, smallest model first; stops at the first model
    that hears the script exactly (no need to ask the bigger ones)."""
    wav = extract_audio(src, trim)
    if wav is None:
        return None
    try:
        hyps = {}
        for mdl in models:
            hyps[mdl] = asr(wav, mdl)
            if script and asr_equal(hyps[mdl], script):
                break
        return hyps
    finally:
        wav.unlink(missing_ok=True)


def _canon(s: str) -> str:
    s = norm_words(american(s)).replace("o'clock", "oclock")
    return s


def judge(script: str, hyps: dict[str, str] | None) -> dict:
    """The reference is always a published script; Whisper only confirms the speaker said it.
    match       a model hears the script word for word
    near_match  the closest model differs by one substituted word, nothing missing or extra (an
                accent the ASR mishears, e.g. "oily bag" for "oily rag"); kept, flagged
    deviates    anything else: words missing or added (the actor ad-libbed, repeated, was cut off)."""
    import jiwer
    if hyps is None:
        return {"status": "no_audio"}
    best = None
    for mdl, h in hyps.items():
        if asr_equal(h, script):
            return {"status": "match", "model": mdl, "asr": hyps}
        o = jiwer.process_words(_canon(script), _canon(h) or "<empty>")
        key = (o.deletions + o.insertions, o.substitutions)
        if best is None or key < best[0]:
            best = (key, mdl)
    (indel, subs), mdl = best
    if indel == 0 and subs <= 1:
        return {"status": "near_match", "model": mdl, "asr": hyps}
    return {"status": "deviates", "asr": hyps}


def check_transcript(src: Path, clip: Clip, models: list[str]) -> dict:
    return judge(clip.transcript, transcribe(src, clip.trim, models, clip.transcript))


# ── build ────────────────────────────────────────────────────────────────────────────────────────

DROPPED: list[dict] = []  # clips the build tried and rejected (no face, transcript mismatch)


def write_attribution(out: Path, rows: list[dict]) -> None:
    lines = ["raw_eval_v2: real-face eval clips (H.264 mp4, CFR at source fps, no audio) + same-stem .txt",
             "transcripts. Built by ml/scripts/build_eval_v2.py; per-clip source, licence and attributes in",
             "manifest.json. Private eval only: do not commit or redistribute; respect each licence below.",
             f"None of these speakers is in the model's training data ({TRAINED_ON}).", ""]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    for key, rs in by.items():
        s = SOURCES[key]
        where = f"{key}/ ({s.access}; {s.terms_note})" if s.gated else "top level"
        lines += [f"{s.name} - {s.licence} - {where}", f"  {s.cite}; {s.home}",
                  f"  {len(rs)} clips, {len({r['speaker'] for r in rs})} speakers: "
                  + ", ".join(r["id"] for r in rs), ""]
    (out / "ATTRIBUTION.txt").write_text("\n".join(lines))
    for key in by:
        s = SOURCES[key]
        if s.gated:
            (out / key / "TERMS.txt").write_text(
                f"{s.name}: {s.licence}\n{s.cite}\n{s.home}\n{s.terms_note}\n"
                f"Access: {s.access}. Keep these clips in this private repo: no redistribution, "
                "evaluation only.\n")


def save_previews(out: Path, cid: str, crop, thumb) -> None:
    import cv2
    d = out / "_previews"
    d.mkdir(parents=True, exist_ok=True)
    if crop is not None:
        cv2.imwrite(str(d / f"{cid}.crop.png"), crop)
    if thumb is not None:
        cv2.imwrite(str(d / f"{cid}.face.jpg"), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def make_row(clip: Clip, src: Source, rel: str, video: dict, check: dict) -> dict:
    return {
        "id": clip.id, "file": rel, "transcript": clip.transcript,
        "words": len(norm_words(clip.transcript).split()),
        "source": src.key, "source_name": src.name, "licence": src.licence, "cite": src.cite,
        "source_url": clip.url, "access": src.access,
        "speaker": clip.speaker, "sex": clip.sex, "age": clip.age, "race": clip.race,
        "ethnicity": clip.ethnicity, **clip.extra, "video": video, "transcript_check": check,
    }


def build_source(src: Source, out: Path, seed: int, asr_models: list[str], dry: bool,
                 limit: int | None) -> list[dict]:
    from lipread.preprocess import MouthCropper
    cands = src.candidates()
    rng = random.Random(f"{seed}:{src.key}")  # per source: adding a source never reshuffles another
    picks = src.pick(cands, rng)[:limit]
    print(f"── {src.name}: {len(cands)} candidates → {len(picks)} picks, "
          f"{len({c.speaker for c in picks})} speakers", flush=True)
    if dry:
        for c in picks:
            print(f"  {c.id:28s} {c.sex} {c.age} {c.race:18s} {c.transcript}")
        return []
    spare: dict[str, list[Clip]] = defaultdict(list)  # same speaker, other items: fallbacks
    taken = {c.id for c in picks}
    for c in cands:
        if c.id not in taken:
            spare[c.speaker].append(c)
    for v in spare.values():
        rng.shuffle(v)
    cropper = MouthCropper()
    raw = CACHE / src.key
    rows = []
    used = {c.id for c in picks}
    for clip in picks:
        tries = [clip] + [c for c in spare[clip.speaker] if c.id not in used][:3]
        for cand in tries:
            used.add(cand.id)
            try:
                srcfile = cand.fetch(raw)
            except Exception as e:  # noqa: BLE001  a dead URL just means: try the next item
                print(f"  {cand.id}: fetch failed ({e})")
                continue
            if cand.extra.get("take_check"):  # MEAD: checked once per take, shared by its views
                check = cand.extra.pop("take_check")
            elif asr_models:
                check = check_transcript(srcfile, cand, asr_models)
            else:
                check = {"status": "skipped"}
            if check["status"] not in ("match", "near_match"):
                print(f"  {cand.id}: transcript check {check['status']}: {check.get('asr')}")
                DROPPED.append({"id": cand.id, "source": src.key, "reason": f"transcript {check['status']}",
                                "asr": check.get("asr")})
                continue
            rel = f"{src.key}/{cand.id}.mp4" if src.gated else f"{cand.id}.mp4"
            dst = out / rel
            video = normalise(srcfile, dst, cand.trim)
            meas, crop, thumb = measure(dst, cropper)
            if meas.get("face_cov", 0) == 0:
                print(f"  {cand.id}: no face ({meas.get('error')}), dropped")
                DROPPED.append({"id": cand.id, "source": src.key, "reason": "no face (MouthCropper)",
                                "detail": meas.get("error"), **{k: cand.extra[k] for k in ("view",) if k in cand.extra}})
                dst.unlink(missing_ok=True)
                continue
            dst.with_suffix(".txt").write_text(cand.transcript + "\n")
            save_previews(out, cand.id, crop, thumb)
            row = make_row(cand, src, rel, video, check)
            row["measured"] = meas
            row["groups"] = groups(row)
            rows.append(row)
            print(f"  {cand.id:28s} {check['status']:9s} iod {meas['iod_px']:5.1f}px yaw {meas['yaw']:+.2f} "
                  f"ita {meas['ita']} luma {meas['face_luma']} → {row['groups']['skin_tone']}/"
                  f"{row['groups']['lighting']}/{row['groups']['pose']}", flush=True)
            break
        else:
            print(f"  {clip.speaker}: no usable clip after {len(tries)} tries")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sources", default=",".join(SOURCES), help=f"comma list of {', '.join(SOURCES)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="max clips per source (quick tests)")
    ap.add_argument("--asr", default="small.en,medium.en,large-v3-turbo",
                    help="Whisper models for the transcript check, tried in order ('' = skip)")
    ap.add_argument("--dry-run", action="store_true", help="print the picks, fetch nothing")
    ap.add_argument("--measure-only", action="store_true", help="recompute measured attributes")
    ap.add_argument("--regroup", action="store_true",
                    help="only recompute groups (e.g. after editing annotations.json)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mpath = a.out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {"clips": []}
    keys = [] if a.regroup else [k for k in a.sources.split(",") if k]
    unknown = set(keys) - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources: {', '.join(sorted(unknown))}")

    if a.measure_only:
        from lipread.preprocess import MouthCropper
        cropper = MouthCropper()
        for r in manifest["clips"]:
            r["measured"], crop, thumb = measure(a.out / r["file"], cropper)
            r["groups"] = groups(r)
            save_previews(a.out, r["id"], crop, thumb)
    else:
        asr_models = [m for m in a.asr.split(",") if m]
        ASR_MODELS[:] = asr_models
        kept = [r for r in manifest["clips"] if r["source"] not in keys]
        new = []
        for k in keys:
            new += build_source(SOURCES[k], a.out, a.seed, asr_models, a.dry_run, a.limit)
        if a.dry_run:
            return
        old_drop = manifest.get("dropped", [])
        if mpath.is_file():  # re-read: another build may have added a source meanwhile
            fresh = json.loads(mpath.read_text())
            kept = [r for r in fresh["clips"] if r["source"] not in keys]
            old_drop = fresh.get("dropped", [])
        manifest["clips"] = kept + new
        manifest["dropped"] = [d for d in old_drop if d["source"] not in keys] + DROPPED

    rows = sorted(manifest["clips"], key=lambda r: r["id"])
    ann_path = a.out / "annotations.json"  # private, next to the clips (not in git)
    skin = json.loads(ann_path.read_text())["skin_tone"] if ann_path.is_file() else {}
    for r in rows:
        r["groups"] = groups(r, skin)
    manifest.update({
        "name": "raw_eval_v2", "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "builder": "ml/scripts/build_eval_v2.py", "seed": a.seed, "trained_on": TRAINED_ON,
        "n_clips": len(rows), "n_speakers": len({r["speaker"] for r in rows}),
        "n_words": sum(r["words"] for r in rows), "clips": rows,
    })
    mpath.write_text(json.dumps(manifest, indent=1))
    write_attribution(a.out, rows)
    cnt = lambda k: dict(Counter(r["groups"][k] for r in rows))  # noqa: E731
    print(f"\n{len(rows)} clips, {manifest['n_speakers']} speakers, {manifest['n_words']} words → {a.out}")
    for k in ("source", "sex", "age_band", "skin_tone", "lighting", "pose", "crop", "motion"):
        print(f"  {k:10s} {cnt(k)}")


if __name__ == "__main__":
    main()
