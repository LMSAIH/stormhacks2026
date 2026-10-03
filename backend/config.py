import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
API_PORT = int(os.getenv("API_PORT", "5000"))
TERMINATOR = "\\x"
SAMPLE_RATE = 8000
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
INITIAL_DEFAULT_VOICE_ID = os.getenv("DEFAULT_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
FRONTEND_ORIGINS = [
	origin.strip()
	for origin in os.getenv(
		"FRONTEND_ORIGINS",
		"http://localhost:5173,http://127.0.0.1:5173",
	).split(",")
	if origin.strip()
]
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true"
DATABASE_URL = os.getenv("TIMESCALE_SERVICE_URL") or os.getenv("DATABASE_URL")