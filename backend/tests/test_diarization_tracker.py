import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except ImportError:  # numpy is only installed with requirements-diarization.txt
    np = None

if np is not None:
    from diarization.config import DiarizationConfig
    from diarization.tracker import SpeakerTracker

SR = 16000
LEVELS = {"A": 0.2, "B": 0.4, "C": 0.6, "D": 0.8}


def pcm(level: float, seconds: float) -> bytes:
    n = int(seconds * SR) // 512 * 512
    return (np.full(n, level * 32767, dtype="<i2")).tobytes()


def silence(seconds: float) -> bytes:
    return pcm(0.0, seconds)


def fake_vad(frame) -> bool:
    return float(np.abs(frame).mean()) > 0.05


def fake_embedder(wav):
    """Histogram of which 'voice level' each sample is near, so a mixed window is a mixed vector."""
    vec = np.zeros(4, dtype=np.float32)
    for i, level in enumerate(LEVELS.values()):
        vec[i] = float((np.abs(wav - level) < 0.05).mean())
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def make(**kw):
    cfg = DiarizationConfig(**kw)
    return SpeakerTracker(cfg, vad=fake_vad, embedder=fake_embedder)


@unittest.skipIf(np is None, "numpy not installed")
class TrackerTests(unittest.TestCase):
    def test_single_speaker_gets_one_label(self):
        t = make()
        t.feed(pcm(LEVELS["A"], 2.0) + silence(1.0))
        self.assertEqual([s.speaker for s in t.segments], ["speaker_0"])
        seg = t.segments[0]
        self.assertAlmostEqual(seg.start, 0.0, places=1)
        self.assertAlmostEqual(seg.end, 2.0, delta=0.1)

    def test_separate_speakers_and_returning_speaker(self):
        t = make()
        t.feed(pcm(LEVELS["A"], 1.5) + silence(1.0))
        t.feed(pcm(LEVELS["B"], 1.5) + silence(1.0))
        t.feed(pcm(LEVELS["A"], 1.5) + silence(1.0))
        self.assertEqual([s.speaker for s in t.segments], ["speaker_0", "speaker_1", "speaker_0"])
        self.assertEqual(t.speaker_count, 2)

    def test_short_utterance_inherits_previous_speaker(self):
        t = make()
        t.feed(pcm(LEVELS["A"], 1.5) + silence(1.0))
        t.feed(pcm(LEVELS["B"], 0.4) + silence(1.0))  # "yeah": too short for a voiceprint
        self.assertEqual([s.speaker for s in t.segments], ["speaker_0", "speaker_0"])
        self.assertTrue(t.segments[1].provisional)

    def test_noise_blip_is_ignored(self):
        t = make()
        t.feed(pcm(LEVELS["A"], 1.5) + silence(1.0))
        t.feed(pcm(LEVELS["B"], 0.1) + silence(1.0))
        self.assertEqual(len(t.segments), 1)

    def test_speaker_change_without_a_pause(self):
        t = make()
        t.feed(pcm(LEVELS["A"], 2.0) + pcm(LEVELS["B"], 3.0) + silence(1.0))
        self.assertEqual([s.speaker for s in t.segments], ["speaker_0", "speaker_1"])
        a, b = t.segments
        self.assertLessEqual(a.end, b.start + 1e-6)
        self.assertAlmostEqual(b.start, 2.0, delta=1.6)  # change is detected a little late

    def test_speaker_at_maps_times_to_labels(self):
        t = make()
        t.feed(pcm(LEVELS["A"], 1.5) + silence(1.0) + pcm(LEVELS["B"], 1.5) + silence(1.0))
        self.assertEqual(t.speaker_at(0.5), "speaker_0")
        self.assertEqual(t.speaker_at(3.0), "speaker_1")
        self.assertIsNone(t.speaker_at(50.0))

    def test_speaker_cap_folds_extra_voices_into_closest(self):
        t = make(max_speakers=2)
        for who in "ABCD":
            t.feed(pcm(LEVELS[who], 1.5) + silence(1.0))
        self.assertEqual(t.speaker_count, 2)

    def test_chunked_feed_matches_single_feed(self):
        data = pcm(LEVELS["A"], 1.5) + silence(1.0) + pcm(LEVELS["B"], 1.5) + silence(1.0)
        one = make()
        one.feed(data)
        many = make()
        for i in range(0, len(data), 1000):  # odd chunk size, not frame aligned
            many.feed(data[i:i + 1000])
        self.assertEqual([s.speaker for s in one.segments], [s.speaker for s in many.segments])

    def test_current_speaker_is_none_until_someone_is_identified(self):
        t = make()
        self.assertIsNone(t.current_speaker)
        t.feed(pcm(LEVELS["A"], 1.0))
        self.assertEqual(t.current_speaker, "speaker_0")


@unittest.skipIf(np is None, "numpy not installed")
class DiarizeWebSocketTests(unittest.TestCase):
    def run_stream(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        import diarization.tracker as tracker_module
        from main import api

        data = pcm(LEVELS["A"], 1.5) + silence(1.0) + pcm(LEVELS["B"], 1.5) + silence(1.0)
        with patch.dict(os.environ, {"DIARIZATION": "1"}), \
                patch.object(tracker_module, "SpeakerTracker", lambda: make()), \
                TestClient(api) as c, c.websocket_connect("/ws/diarize") as ws:
            self.assertEqual(ws.receive_json(), {"type": "ready"})
            for i in range(0, len(data), 3200):
                ws.send_bytes(data[i:i + 3200])
            ws.send_text("end")
            msgs = []
            while True:
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] == "done":
                    return msgs

    def test_streams_segments_then_done_with_final_timeline(self):
        msgs = self.run_stream()
        self.assertTrue(any(m["type"] == "segment" and m["speaker"] == "speaker_0" for m in msgs))
        done = msgs[-1]
        self.assertEqual(done["speakers"], 2)
        self.assertEqual([s["speaker"] for s in done["segments"]], ["speaker_0", "speaker_1"])

    def test_closed_when_disabled(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from main import api

        with patch.dict(os.environ, {"DIARIZATION": "0"}), TestClient(api) as c, \
                c.websocket_connect("/ws/diarize") as ws:
            self.assertEqual(ws.receive()["code"], 4404)


if __name__ == "__main__":
    unittest.main()
