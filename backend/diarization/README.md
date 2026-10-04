# diarization (optional, backend only)

Speaker labels for the listening panel. Text comes from ElevenLabs Scribe Realtime
(`backend/stt/`); this package only decides *which speaker* said it: VAD -> voiceprint
(Resemblyzer) -> online clustering. Speakers are anonymous (`speaker_0`, `speaker_1`, ...); the
client renames them. The frontend is not touched.

## Turn on

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-diarization.txt && .venv/bin/pip install resemblyzer --no-deps
echo DIARIZATION=1 >> .env
```

(Tested on Python 3.11; torch/librosa wheels for 3.14 may not exist yet. mp3/m4a decoding goes
through librosa and uses ffmpeg when needed.) Tunables are `DIARIZATION_*` env vars, see `config.py`.

## Test it the way the client will use it (WebSocket)

`/ws/diarize` takes streamed PCM frames and sends back speaker segments as they are decided.
With the server running (`DIARIZATION=1 python main.py`):

```
.venv/bin/python tests/diarization_stream_test.py meeting.mp3            # real-time pacing
.venv/bin/python tests/diarization_stream_test.py meeting.mp3 --speed 0  # as fast as possible
.venv/bin/python tests/diarization_stream_test.py meeting.mp3 --mode stt # full path incl. ElevenLabs text
```

Writes `tests/diarization_output/<name>-<mode>.txt`: live events with latency, the final speaker
timeline, and a lane chart (one row per speaker) to eyeball whether labels match.

## Test it with a file (HTTP / in-process)

Server running (`python main.py` with `DIARIZATION=1`):

```
.venv/bin/python -m diarization.tools.post_file meeting.mp3
.venv/bin/python -m diarization.tools.post_file meeting.mp3 -p threshold=0.7 -p max_speakers=3
# or plain curl:
curl -X POST --data-binary @meeting.mp3 "http://localhost:5000/api/diarize?format=mp3"
```

No server? Same code in-process: `.venv/bin/python -m diarization.tools.label meeting.mp3 --threshold 0.72`
(also prints per-voiceprint latency).

`POST /api/diarize` takes the raw file as the request body and returns
`{duration, speakers, segments: [{speaker, start, end, provisional}], timing}`. It is 404 unless
`DIARIZATION=1`, and has no login on purpose (local testing aid).

Measured on a 20 s four-person test clip (macOS `say` voices) on an Apple laptop: 3 speakers
found, voiceprint ~8 ms median, ~100x real time. Real rooms will be harder; tune `threshold`.

## Live path (later)

`/ws/stt` (`backend/api/stt.py`, `backend/stt/`) streams mic PCM to Scribe Realtime and, with
`DIARIZATION=1`, labels each caption using this package via `create_live_diarizer()`.
`diarization.get_tts_gate(user_id)` can drop the app's own TTS voice from captions; hooking it into
`websocket_server.py` (call `gate.note_text(text)` and `gate.note_audio(audio_seconds(chunk))`) is
not done yet.

## Remove it completely

1. `rm -r backend/diarization backend/requirements-diarization.txt`
2. `backend/main.py`: delete the `diarize_router` import and `include_router` line.
3. `backend/api/stt.py`: delete the `diarization` import, set `diarizer, gate = None, None`
   (or delete `api/stt.py`, `stt/` and their `main.py` lines to drop live captions too).
4. Tests: `tests/test_diarization_tracker.py`, and the merge/gate/endpoint tests in `tests/test_stt_session.py`.

## Known limits

- Scribe word timestamps are assumed to be seconds since the first audio sample of the session;
  the live client must stream audio continuously (silence included).
- First 10-20 s of a conversation are the least accurate while voiceprints form; very short
  utterances ("yeah") inherit the previous speaker.
