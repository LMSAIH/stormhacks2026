"""Backend registry. Add a backend = one new module + one line here."""

from importlib import import_module

from tts.base import TtsBackend, TtsConfig

# name -> (module, class). Imported lazily so one backend's deps never break another.
_REGISTRY = {
    "flash": ("tts.flash_ws", "FlashWsBackend"),
    "v4": ("tts.v4_ws", "V4DialogueWsBackend"),
}


def available() -> list[str]:
    return list(_REGISTRY)


def create_backend(name: str, config: TtsConfig) -> TtsBackend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; choose from {available()}")
    module, cls = _REGISTRY[name]
    return getattr(import_module(module), cls)(config)
