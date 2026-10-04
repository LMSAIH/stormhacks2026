import base64
import json
from http.cookies import SimpleCookie

from itsdangerous import BadSignature, TimestampSigner

from config import SESSION_SECRET


SESSION_COOKIE = "voice_session"
SESSION_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
_signer = TimestampSigner(str(SESSION_SECRET), salt="starlette.sessions")


def get_user_from_cookie_header(cookie_header: str) -> dict | None:
	cookies = SimpleCookie()
	try:
		cookies.load(cookie_header)
		cookie = cookies.get(SESSION_COOKIE)
		if cookie is None:
			return None
		payload = _signer.unsign(cookie.value, max_age=SESSION_MAX_AGE_SECONDS)
		session = json.loads(base64.b64decode(payload))
	except (BadSignature, ValueError, json.JSONDecodeError):
		return None

	user = session.get("user")
	return user if isinstance(user, dict) and user.get("id") else None