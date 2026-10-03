import asyncio
import base64
import io
import json
import unittest
import wave
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi.responses import RedirectResponse
from itsdangerous import TimestampSigner
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from api import auth as auth_routes
from api import chat as chat_routes
from api import voices as voice_routes
from config import SAMPLE_RATE, SESSION_SECRET
from main import api
from services import elevenlabs
from state import get_default_voice_id, set_default_voice_id
import websocket_server
from websocket_server import handle_connection


def make_session_cookie() -> str:
    session = {"user": {"id": "test-user", "email": "test@example.com"}}
    payload = base64.b64encode(json.dumps(session).encode("utf-8"))
    signer = TimestampSigner(str(SESSION_SECRET), salt="starlette.sessions")
    return signer.sign(payload).decode("utf-8")


class WebSocketHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffers_text_until_terminator_then_returns_audio(self):
        async with serve(handle_connection, "127.0.0.1", 0, compression=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}",
                additional_headers={"Cookie": f"voice_session={make_session_cookie()}"},
            ) as websocket:
                with patch.object(
                    websocket_server,
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

    async def test_rejects_websocket_without_session(self):
        async with serve(handle_connection, "127.0.0.1", 0, compression=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                with self.assertRaises(Exception):
                    await websocket.recv()

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
        with (
            patch.object(auth_routes, "GOOGLE_CLIENT_ID", "test-client-id"),
            patch.object(auth_routes, "GOOGLE_CLIENT_SECRET", "test-client-secret"),
            patch.object(
                auth_routes.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": "test-user",
                        "email": "test@example.com",
                        "email_verified": True,
                    },
                }),
            ),
        ):
            response = self.client.get(
                "/api/auth/google/callback?code=test&state=test",
                follow_redirects=False,
            )
        if response.status_code != 303:
            raise AssertionError(f"Test sign-in failed: {response.status_code}")
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

    def test_voice_routes_require_sign_in(self):
        self.client.cookies.clear()

        response = self.client.get("/api/voices")

        self.assertEqual(response.status_code, 401)


class GoogleAuthApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api)
        self.client_id_patch = patch.object(auth_routes, "GOOGLE_CLIENT_ID", "test-client-id")
        self.client_secret_patch = patch.object(
            auth_routes,
            "GOOGLE_CLIENT_SECRET",
            "test-client-secret",
        )
        self.client_id_patch.start()
        self.client_secret_patch.start()

    def tearDown(self):
        self.client_secret_patch.stop()
        self.client_id_patch.stop()
        self.client.close()

    def test_login_redirect_uses_registered_callback(self):
        redirect = RedirectResponse("https://accounts.google.com/o/oauth2/auth")
        with patch.object(
            auth_routes.oauth.google,
            "authorize_redirect",
            new=AsyncMock(return_value=redirect),
        ) as authorize_redirect:
            response = self.client.get("/api/auth/google/login", follow_redirects=False)

        self.assertEqual(response.status_code, 307)
        callback_url = authorize_redirect.await_args.args[1]
        self.assertEqual(
            str(callback_url),
            "http://testserver/api/auth/google/callback",
        )

    def test_callback_creates_session_and_logout_clears_it(self):
        userinfo = {
            "sub": "google-user-1",
            "email": "person@example.com",
            "email_verified": True,
            "name": "Example Person",
            "picture": "https://example.com/avatar.png",
        }
        with patch.object(
            auth_routes.oauth.google,
            "authorize_access_token",
            new=AsyncMock(return_value={"userinfo": userinfo}),
        ):
            response = self.client.get(
                "/api/auth/google/callback?code=test&state=test",
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "http://localhost:5173")
        self.assertEqual(
            self.client.get("/api/auth/me").json()["user"]["id"],
            "google-user-1",
        )

        logout_response = self.client.post("/api/auth/logout")

        self.assertEqual(logout_response.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_unverified_google_identity_is_rejected(self):
        with patch.object(
            auth_routes.oauth.google,
            "authorize_access_token",
            new=AsyncMock(return_value={
                "userinfo": {"sub": "google-user-1", "email_verified": False},
            }),
        ):
            response = self.client.get(
                "/api/auth/google/callback?code=test&state=test",
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 401)


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api)
        self.sign_in(self.client, "chat-user-1")

    def tearDown(self):
        self.client.close()

    @staticmethod
    def sign_in(client, user_id):
        with (
            patch.object(auth_routes, "GOOGLE_CLIENT_ID", "test-client-id"),
            patch.object(auth_routes, "GOOGLE_CLIENT_SECRET", "test-client-secret"),
            patch.object(
                auth_routes.oauth.google,
                "authorize_access_token",
                new=AsyncMock(return_value={
                    "userinfo": {
                        "sub": user_id,
                        "email": f"{user_id}@example.com",
                        "email_verified": True,
                    },
                }),
            ),
        ):
            response = client.get(
                "/api/auth/google/callback?code=test&state=test",
                follow_redirects=False,
            )
        if response.status_code != 303:
            raise AssertionError(f"Test sign-in failed: {response.status_code}")

    def test_saves_and_retrieves_full_chat_json(self):
        chat = {
            "speakers": [
                {"id": "tom", "name": "Tom"},
                {"id": "paul", "name": "Paul"},
            ],
            "messages": [
                {"speaker_id": "tom", "text": "Hello"},
                {"speaker_id": "paul", "text": "Back message"},
                {"speaker_id": "tom", "text": "Another message"},
            ],
        }

        with (
            patch.object(chat_routes, "save_chat", new=AsyncMock()) as save,
            patch.object(chat_routes, "get_chat", new=AsyncMock(return_value=chat)),
        ):
            save_response = self.client.put("/api/chat", json=chat)
            get_response = self.client.get("/api/chat")

        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(save_response.json(), {"saved": True})
        save.assert_awaited_once_with("chat-user-1", chat["speakers"], chat["messages"])
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json(), chat)

    def test_rejects_messages_with_unknown_speaker_id(self):
        response = self.client.put(
            "/api/chat",
            json={
                "speakers": [{"id": "tom", "name": "Tom"}],
                "messages": [{"speaker_id": "paul", "text": "Unknown speaker"}],
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_data_is_scoped_to_user(self):
        saved_chats = {
            "chat-user-1": {
                "speakers": [{"id": "tom", "name": "Tom"}],
                "messages": [{"speaker_id": "tom", "text": "Private chat"}],
            },
        }
        with patch.object(
            chat_routes,
            "get_chat",
            new=AsyncMock(side_effect=lambda user_id: saved_chats.get(user_id)),
        ):
            own_chat_response = self.client.get("/api/chat")

        other_user_client = TestClient(api)
        self.addCleanup(other_user_client.close)
        self.sign_in(other_user_client, "chat-user-2")

        with patch.object(
            chat_routes,
            "get_chat",
            new=AsyncMock(side_effect=lambda user_id: saved_chats.get(user_id)),
        ):
            response = other_user_client.get("/api/chat")

        self.assertEqual(own_chat_response.status_code, 200)
        self.assertEqual(own_chat_response.json(), saved_chats["chat-user-1"])
        self.assertEqual(response.status_code, 404)

    def test_chat_endpoints_require_sign_in(self):
        self.client.cookies.clear()

        self.assertEqual(self.client.get("/api/chat").status_code, 401)
        self.assertEqual(
            self.client.put(
                "/api/chat",
                json={"speakers": [], "messages": []},
            ).status_code,
            401,
        )


if __name__ == "__main__":
    unittest.main()