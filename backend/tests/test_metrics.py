import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.metrics import Recorder, aggregate, format_table, write_csv  # noqa: E402
from tts.base import AudioChunk  # noqa: E402


def chunk(seg, t, secs=0.5, final=False, empty=False):
    n = 0 if empty else int(secs * 24000) * 2
    return AudioChunk(b"\0" * n, 24000, "pcm", seg, t, final)


def build():
    r = Recorder()
    r.note_segment(0, 0.0, 0.1)
    r.note_segment(1, 1.0, 1.1)
    r.note_chunk(chunk(0, 0.4))
    r.note_chunk(chunk(0, 0.6))
    r.note_chunk(chunk(0, 1.0))  # gap 0.4, audio 1.5s, playback ends 1.9
    r.note_chunk(chunk(0, 5, final=True))
    r.note_chunk(chunk(0, 6, empty=True))
    r.note_chunk(chunk(1, 2.4))  # handoff gap 0.5
    return r


def test_summary():
    r = build()
    s = r.segments()
    assert s[0].n_chunks == 3 and abs(s[0].audio_seconds - 1.5) < 1e-9
    assert abs(s[0].max_chunk_gap - 0.4) < 1e-9
    m = r.summarize()
    p = m["per_segment"]
    assert abs(p[0]["ttfa"] - 0.3) < 1e-9
    assert abs(p[0]["end_to_end"] - 0.4) < 1e-9
    assert abs(p[0]["gen_speed"] - 1.5 / 0.6) < 1e-9
    assert p[1]["gen_speed"] is None
    assert p[0]["handoff_gap"] is None
    assert abs(p[1]["handoff_gap"] - 0.5) < 1e-9
    assert "connect_time" not in m
    r2 = Recorder()
    r2.note_segment(0, 0.5, 0.6, t_first_word=0.1, n_words=3, text="a b c", reason="pause")
    t = r2.segments()[0]
    assert (t.t_first_word, t.n_words, t.text, t.reason) == (0.1, 3, "a b c", "pause")


def test_no_pause_when_buffered():
    r = Recorder()
    r.note_segment(0, 0, 0)
    r.note_segment(1, 0, 0)
    r.note_chunk(chunk(0, 1.0, 2.0))
    r.note_chunk(chunk(1, 1.5, 1.0))
    assert r.summarize()["per_segment"][1]["handoff_gap"] == 0.0


def test_aggregate_and_table(tmp_path=None):
    runs = [build().summarize(), build().summarize()]
    a = aggregate(runs)
    assert aggregate([runs[0]])["ttfa_mean"]["p95"] == runs[0]["ttfa_mean"]
    t = format_table({"a": a, "b": a})
    assert "a typical" in t and "TTS wait, first" in t
    d = tmp_path or tempfile.mkdtemp()
    p = os.path.join(str(d), "o.csv")
    write_csv(p, [{k: v for k, v in x.items() if k != "per_segment"} for x in runs])
    assert open(p).read().startswith("Segments sent")


if __name__ == "__main__":
    test_summary()
    test_no_pause_when_buffered()
    test_aggregate_and_table()
    print("ok")
