from config import INITIAL_DEFAULT_VOICE_ID


_default_voice_id = INITIAL_DEFAULT_VOICE_ID


def get_default_voice_id() -> str:
	return _default_voice_id


def set_default_voice_id(voice_id: str) -> str:
	global _default_voice_id
	_default_voice_id = voice_id
	return _default_voice_id