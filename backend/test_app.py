import asyncio
import io
import unittest
import wave
from unittest.mock import AsyncMock, patch

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import app


class WebSocketHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffers_text_until_terminator_then_returns_audio(self):
        async with serve(app.handle_connection, "127.0.0.1", 0, compression=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                with patch.object(
                    app,
                    "callElevenLabs",
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
                    call_eleven_labs.assert_awaited_once_with("The quick brown fox")

    async def test_dummy_audio_is_a_valid_wav(self):
        audio = await app.callElevenLabs("A test sentence")

        self.assertEqual(audio[:4], b"RIFF")
        self.assertEqual(audio[8:12], b"WAVE")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), app.SAMPLE_RATE)
            self.assertGreater(wav.getnframes(), 0)


if __name__ == "__main__":
    unittest.main()