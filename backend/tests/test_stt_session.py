import asyncio
import base64
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402
from itsdangerous import TimestampSigner  # noqa: E402

from api import stt as stt_route  # noqa: E402
from config import SESSION_SECRET  # noqa: E402
from diarization.merge import Piece, split_by_speaker  # noqa: E402
from diarization.tts_gate import TtsGate  # noqa: E402
from main import api  # noqa: E402
from stt.session import SttSession  # noqa: E402


def words(*items):
    """items: (text, start, end) -> Scribe style tokens with spacing between words."""
    out = []
    for i, (text, start, end) in enumerate(items):
        if i:
            out.append({"text": " ", "start": start, "end": start, "type": "spacing"})
        out.append({"text": text, "start": start, "end": end, "type": "word"})
    return out


class MergeTests(unittest.TestCase):
    def lookup(self, table):
        return lambda t: next((s for a, b, s in table if a <= t < b), None)

    def test_splits_a_commit_at_the_speaker_change(self):
        w = words(("yeah", 0.0, 0.4), ("I", 0.5, 0.6), ("agree", 0.6, 1.0), ("ok", 2.0, 2.3), ("great", 2.4, 2.9))
        pieces = split_by_speaker(w, self.lookup([(0, 1.5, "speaker_0"), (1.5, 9, "speaker_1")]))
        self.assertEqual(pieces, [Piece("speaker_0", "yeah I agree"), Piece("speaker_1", "ok great")])

    def test_single_word_island_is_smoothed_away(self):
        w = words(("a", 0.0, 0.2), ("b", 0.3, 0.5), ("c", 0.6, 0.8))
        pieces = split_by_speaker(w, self.lookup([(0, 0.25, "x"), (0.25, 0.55, "y"), (0.55, 9, "x")]))
        self.assertEqual(pieces, [Piece("x", "a b c")])

    def test_unlabelled_words_take_the_neighbouring_speaker(self):
        w = words(("hi", 0.0, 0.2), ("there", 5.0, 5.2))
        pieces = split_by_speaker(w, self.lookup([(0, 1, "speaker_0")]))
        self.assertEqual(pieces, [Piece("speaker_0", "hi there")])

    def test_no_speakers_known_gives_one_unlabelled_piece(self):
        pieces = split_by_speaker(words(("hello", 0, 1)), lambda t: None)
        self.assertEqual(pieces, [Piece(None, "hello")])

    def test_languages_without_spacing_tokens_join_with_spaces(self):
        w = [{"text": "你好", "start": 0, "end": 1, "type": "word"}, {"text": "吗", "start": 1, "end": 2, "type": "word"}]
        self.assertEqual(split_by_speaker(w, lambda t: "s")[0].text, "你好 吗")


class GateTests(unittest.TestCase):
    def make(self):
        self.t = 100.0
        return TtsGate(now=lambda: self.t)

    def test_matching_text_during_playback_is_echo(self):
        g = self.make()
        g.note_text("The quick brown fox jumps over the lazy dog")
        g.note_audio(3.0)
        self.assertTrue(g.is_echo("the quick brown fox jumps over", 100.0, 102.0))

    def test_different_speech_during_playback_is_kept(self):
        g = self.make()
        g.note_text("The quick brown fox jumps over the lazy dog")
        g.note_audio(3.0)
        self.assertFalse(g.is_echo("what time is the meeting tomorrow", 100.0, 102.0))

    def test_weak_match_outside_playback_window_is_kept(self):
        g = self.make()
        g.note_text("I would like some coffee please")
        g.note_audio(2.0)
        self.t = 200.0
        self.assertFalse(g.is_echo("some coffee would be nice", 199.0, 200.0))

    def test_exact_repeat_is_echo_even_after_playback(self):
        g = self.make()
        g.note_text("see you tomorrow at noon")
        g.note_audio(2.0)
        self.t = 110.0
        self.assertTrue(g.is_echo("see you tomorrow at noon", 109.0, 110.0))

    def test_audio_chunks_extend_one_window(self):
        g = self.make()
        g.note_audio(1.0)
        g.note_audio(1.0)  # arrives while the first is still "playing"
        self.assertEqual(len(g._windows), 1)
        self.assertAlmostEqual(g._windows[0][1] - g._windows[0][0], 2.0)


class FakeScribe:
    def __init__(self):
        self.audio = []
        self.sent_events = asyncio.Queue()
        self.closed = False

    async def open(self):
        pass

    async def send_audio(self, pcm):
        self.audio.append(pcm)

    async def events(self):
        while True:
            msg = await self.sent_events.get()
            if msg is None:
                return
            yield msg

    async def close(self):
        self.closed = True
        self.sent_events.put_nowait(None)


class FakeDiarizer:
    def __init__(self, pieces=None, speaker="speaker_0"):
        self.fed = []
        self.pieces = pieces or []
        self.speaker = speaker
        self.waited = None

    def feed(self, pcm):
        self.fed.append(pcm)

    def current_speaker(self):
        return self.speaker

    async def wait_for(self, t, timeout=0.8):
        self.waited = t

    def split(self, w):
        return self.pieces


class SessionTests(unittest.IsolatedAsyncioTestCase):
    def make(self, **kw):
        self.out = []

        async def send(e):
            self.out.append(e)

        self.scribe = FakeScribe()
        return SttSession(self.scribe, send, **kw)

    async def test_text_only_mode_streams_partials_then_final_without_speaker_key(self):
        s = self.make()
        await s.on_event({"message_type": "session_started"})
        await s.on_event({"message_type": "partial_transcript", "text": "hello wor"})
        await s.on_event({"message_type": "partial_transcript", "text": ""})  # keepalive
        await s.on_event({"message_type": "committed_transcript", "text": "hello world"})
        await s.on_event({"message_type": "committed_transcript_with_timestamps", "text": "hello world", "words": []})
        self.assertEqual(self.out, [
            {"type": "ready"},
            {"type": "utterance", "id": "u0-0", "text": "hello wor", "final": False},
            {"type": "utterance", "id": "u0-0", "text": "hello world", "final": True},
        ])

    async def test_audio_is_relayed_to_scribe_and_diarizer(self):
        d = FakeDiarizer()
        s = self.make(diarizer=d)
        await s.on_audio(b"\x01\x02")
        self.assertEqual(self.scribe.audio, [b"\x01\x02"])
        self.assertEqual(d.fed, [b"\x01\x02"])

    async def test_diarized_commit_is_split_per_speaker_with_stable_ids(self):
        d = FakeDiarizer(pieces=[Piece("speaker_0", "I agree"), Piece("speaker_1", "great")])
        s = self.make(diarizer=d)
        await s.on_event({"message_type": "partial_transcript", "text": "I agree gre"})
        await s.on_event({"message_type": "committed_transcript", "text": "I agree great"})  # ignored
        await s.on_event({"message_type": "committed_transcript_with_timestamps", "text": "I agree great",
                          "words": words(("I", 0, 0.2), ("agree", 0.3, 0.9), ("great", 2.0, 2.5))})
        self.assertEqual(self.out[0], {"type": "utterance", "id": "u0-0", "text": "I agree gre",
                                       "final": False, "speaker": "speaker_0"})
        self.assertEqual(self.out[1:], [
            {"type": "utterance", "id": "u0-0", "text": "I agree", "final": True, "speaker": "speaker_0"},
            {"type": "utterance", "id": "u0-1", "text": "great", "final": True, "speaker": "speaker_1"},
        ])
        self.assertEqual(d.waited, 2.5)
        await s.on_event({"message_type": "partial_transcript", "text": "next"})
        self.assertEqual(self.out[-1]["id"], "u1-0")

    async def test_echo_of_our_own_tts_is_dropped(self):
        gate = TtsGate()
        gate.note_text("please bring the report tomorrow")
        gate.note_audio(5.0)
        d = FakeDiarizer(pieces=[Piece("speaker_0", "please bring the report tomorrow")])
        s = self.make(diarizer=d, gate=gate)
        await s.on_audio(b"\x00\x00")
        await s.on_event({"message_type": "partial_transcript", "text": "please bring the report"})
        await s.on_event({"message_type": "committed_transcript_with_timestamps",
                          "text": "please bring the report tomorrow",
                          "words": words(("please", 0, 0.3), ("bring", 0.4, 0.6))})
        self.assertEqual(self.out, [{"type": "drop", "id": "u0-0"}])

    async def test_scribe_errors_are_forwarded(self):
        s = self.make()
        await s.on_event({"message_type": "quota_exceeded", "error": "out of credits"})
        self.assertEqual(self.out, [{"type": "error", "message": "out of credits"}])


class RouteTests(unittest.TestCase):
    def cookie(self):
        payload = base64.b64encode(json.dumps({"user": {"id": "u1", "email": "a@b.c"}}).encode())
        return "voice_session=" + TimestampSigner(str(SESSION_SECRET), salt="starlette.sessions").sign(payload).decode()

    def test_rejects_without_session(self):
        with TestClient(api) as c, c.websocket_connect("/ws/stt") as ws:
            self.assertEqual(ws.receive()["code"], 4401)

    def test_streams_audio_to_scribe_and_events_back(self):
        scribe = FakeScribe()
        with patch.object(stt_route, "ScribeRealtime", return_value=scribe), \
                patch.object(stt_route, "create_live_diarizer", return_value=None), \
                TestClient(api) as c, \
                c.websocket_connect("/ws/stt", headers={"Cookie": self.cookie()}) as ws:
            ws.send_bytes(b"\x00\x01" * 160)
            scribe.sent_events.put_nowait({"message_type": "session_started"})
            self.assertEqual(ws.receive_json(), {"type": "ready"})
            scribe.sent_events.put_nowait({"message_type": "committed_transcript", "text": "hi there"})
            self.assertEqual(ws.receive_json(),
                             {"type": "utterance", "id": "u0-0", "text": "hi there", "final": True})
        self.assertEqual(scribe.audio, [b"\x00\x01" * 160])
        self.assertTrue(scribe.closed)


class DiarizeEndpointTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {"DIARIZATION": "0"}), TestClient(api) as c:
            self.assertEqual(c.post("/api/diarize", content=b"abc").status_code, 404)

    def test_returns_service_result_and_forwards_overrides(self):
        fake = {"duration": 1.0, "speakers": 1, "segments": [], "timing": {}}
        with patch.dict(os.environ, {"DIARIZATION": "1"}), \
                patch("diarization.service.diarize_audio", return_value=fake) as svc, \
                TestClient(api) as c:
            r = c.post("/api/diarize?format=mp3&threshold=0.6", content=b"abc")
            bad = c.post("/api/diarize?nonsense=1", content=b"abc")
        self.assertEqual(r.json(), fake)
        svc.assert_called_once_with(b"abc", suffix="mp3", overrides={"threshold": 0.6})
        self.assertEqual(bad.status_code, 422)


if __name__ == "__main__":
    unittest.main()
