"""Live ElevenLabs socket integration test.

Run with RUN_ELEVENLABS_INTEGRATION=1 from backend/ to use the configured API key.
"""

import asyncio
import base64
import json
import os
import unittest

from itsdangerous import TimestampSigner
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from config import SESSION_SECRET, TERMINATOR
from websocket_server import handle_connection


def make_session_cookie() -> str:
    session = {"user": {"id": "elevenlabs-integration-test"}}
    payload = base64.b64encode(json.dumps(session).encode("utf-8"))
    signer = TimestampSigner(str(SESSION_SECRET), salt="starlette.sessions")
    return signer.sign(payload).decode("utf-8")


@unittest.skipUnless(
    os.getenv("RUN_ELEVENLABS_INTEGRATION") == "1",
    "set RUN_ELEVENLABS_INTEGRATION=1 to call ElevenLabs",
)
class ElevenLabsSocketIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sentence_returns_elevenlabs_audio(self):
        async with serve(
            handle_connection,
            "127.0.0.1",
            0,
            compression=None,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}",
                additional_headers={"Cookie": f"voice_session={make_session_cookie()}"},
            ) as websocket:
                await websocket.send(
                    f"This is a live ElevenLabs socket integration test.{TERMINATOR}"
                )
                audio = await asyncio.wait_for(websocket.recv(), timeout=30)

                self.assertIsInstance(audio, bytes, "expected a binary audio frame")
                self.assertGreater(len(audio), 0, "expected non-empty ElevenLabs audio")


if __name__ == "__main__":
    unittest.main()