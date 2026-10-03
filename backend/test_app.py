import asyncio
import io
import unittest
import wave
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from api import voices as voice_routes
from config import SAMPLE_RATE
from main import api
from services import audio as audio_service
from services import elevenlabs
from state import get_default_voice_id, set_default_voice_id
from websocket_server import handle_connection


class WebSocketHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffers_text_until_terminator_then_returns_audio(self):
        async with serve(handle_connection, "127.0.0.1", 0, compression=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                with patch.object(
                    elevenlabs,
                    "generate_speech",
                    new=AsyncMock(return_value=b"mock-audio"),
                ) as call_eleven_labs:
                    await websocket.send("The quick ")
                    await websocket.send("brown fox\\")

                    with self.assertRaises(asyncio.TimeoutError):
                        await asyncio.wait_for(websocket.recv(), timeout=0.05)
                    call_eleven_labs.assert_not_awaited()

                    await websocket.send("x")
                    response = await asyncio.wait_for(websocket.recv(), timeout=1)

                    self.assertEqual(response, b"mock-audio")
                    call_eleven_labs.assert_awaited_once_with(
                        "The quick brown fox",
                        get_default_voice_id(),
                    )

    async def test_dummy_audio_is_a_valid_wav(self):
        audio = await elevenlabs.generate_speech("A test sentence", get_default_voice_id())

        self.assertEqual(audio[:4], b"RIFF")
        self.assertEqual(audio[8:12], b"WAVE")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), SAMPLE_RATE)
            self.assertGreater(wav.getnframes(), 0)


class VoiceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api)
        self.original_voice_id = get_default_voice_id()

    def tearDown(self):
        set_default_voice_id(self.original_voice_id)
        self.client.close()

    def test_lists_available_voices_and_current_default(self):
        voices = [{"voice_id": "voice-123", "name": "Demo Voice"}]
        with patch.object(
            voice_routes,
            "list_available_voices",
            new=AsyncMock(return_value=voices),
        ):
            response = self.client.get("/api/voices")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["voices"], voices)
        self.assertEqual(response.json()["default_voice_id"], self.original_voice_id)

    def test_updates_default_voice(self):
        response = self.client.put("/api/voice", json={"voice_id": "voice-456"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"default_voice_id": "voice-456"})
        self.assertEqual(get_default_voice_id(), "voice-456")

    def test_rejects_blank_voice_id_without_changing_default(self):
        response = self.client.put("/api/voice", json={"voice_id": "   "})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(get_default_voice_id(), self.original_voice_id)


if __name__ == "__main__":
    unittest.main()