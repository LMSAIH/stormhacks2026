import os


HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
API_PORT = 8000
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