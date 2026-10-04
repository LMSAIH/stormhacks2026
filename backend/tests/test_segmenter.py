import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streaming.profiles import SegmenterProfile  # noqa: E402
from streaming.segmenter import Segmenter  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(**kw):
    c = Clock()
    return Segmenter(SegmenterProfile(**kw), c), c


def feed(s, c, words):
    out = []
    for w in words:
        c.t += 0.01
        out += s.push(w)
    return out


def test_uncommitted_ignored():
    s, c = make()
    assert s.push("hi", committed=False) == []
    assert s.end_turn() == []


def test_turn_end_flushes_below_min():
    s, c = make()
    feed(s, c, ["hi"])
    out = s.end_turn()
    assert len(out) == 1 and out[0].is_turn_end and out[0].text == "hi" and out[0].n_words == 1
    assert out[0].reason == "turn_end" and out[0].first_word_t == 0.01


def test_pause_flush_below_min():
    s, c = make(pause_ms=350)
    feed(s, c, ["hi"])
    c.t += 0.3
    assert s.tick() == []
    c.t += 0.06
    out = s.tick()
    assert len(out) == 1 and not out[0].is_turn_end and out[0].t_words_done == 0.01
    assert out[0].reason == "pause"
    assert s.tick() == []


def test_punct_needs_min_words():
    s, c = make(punct_min_words=3, min_words=3)
    assert feed(s, c, ["Hi,", "there"]) == []
    out = feed(s, c, ["friend."])
    assert [x.text for x in out] == ["Hi, there friend."]
    assert out[0].n_words == 3 and out[0].reason == "punctuation" and abs(out[0].first_word_t - 0.01) < 1e-9 and abs(out[0].t_words_done - 0.03) < 1e-9
    assert feed(s, c, ["Ok,", "no"]) == []  # punct mid-buffer doesn't flush


def test_sentence_ender_flush():
    s, c = make()
    assert [x.text for x in feed(s, c, ["a", "b", "c?"])] == ["a b c?"]


def test_max_words():
    s, c = make(max_words=4, max_chars=999)
    out = feed(s, c, list("abcde"))
    assert [x.text for x in out] == ["a b c d"] and out[0].reason == "max_words"


def test_max_chars():
    s, c = make(max_chars=12, max_words=99)
    out = feed(s, c, ["aaaa", "bbbb", "cccc"])
    assert [x.text for x in out] == ["aaaa bbbb cccc"] and out[0].reason == "max_chars"


def test_min_words_respected_for_chars():
    s, c = make(max_chars=5, min_words=3)
    assert feed(s, c, ["longword", "another"]) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
