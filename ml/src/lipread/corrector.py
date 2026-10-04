"""LLM corrector client: the thin transport under the Agentic Condom (`lipread.agentic_condom`).

Talks to any OpenAI-compatible chat endpoint so the same code works for vLLM on the RunPod pod
(`CONDOM=1 ml/runpod/serve.sh`), OpenRouter, or Ollama locally:
  CORRECTOR_BASE_URL  e.g. http://127.0.0.1:8001/v1 (vLLM); unset + CORRECTOR_API_KEY → OpenRouter
  CORRECTOR_MODEL     the served model name
  CORRECTOR_API_KEY   bearer token for hosted providers (server-side only, never in the frontend)
Unconfigured → passthrough, so the pipeline never blocks on it.
"""

from __future__ import annotations

import os
import threading

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "You fix the output of a lip-reading model. The input is a noisy transcript of what a person "
    "silently mouthed; words may be wrong but usually look similar on the lips (e.g. p/b/m, f/v). "
    "Return only the most likely intended sentence in plain English, no quotes, no explanation. "
    "If the input is empty or meaningless, return it unchanged."
)


class Corrector:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 10.0):
        self.api_key = api_key or os.environ.get("CORRECTOR_API_KEY", "")
        default_url = OPENROUTER_URL if self.api_key else ""
        self.base_url = (base_url or os.environ.get("CORRECTOR_BASE_URL", "") or default_url).rstrip("/")
        self.model = model or os.environ.get("CORRECTOR_MODEL", "")
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def _http(self) -> httpx.Client:
        # One pooled client: a hosted provider's TLS handshake per call would eat the 500 ms budget.
        with self._lock:
            if self._client is None:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                self._client = httpx.Client(headers=headers, timeout=self.timeout)
            return self._client

    def complete(self, messages: list[dict], *, max_tokens: int = 128, timeout: float | None = None,
                 stop: list[str] | None = None) -> str:
        """One greedy chat completion → its text. Raises on HTTP errors and timeouts."""
        body: dict = {"model": self.model, "temperature": 0.0, "max_tokens": max_tokens, "messages": messages}
        if stop:
            body["stop"] = stop
        resp = self._http().post(f"{self.base_url}/chat/completions", json=body,
                                 timeout=self.timeout if timeout is None else timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""

    def correct(self, text: str) -> str:
        """Free rewrite of `text`, no confidence gate. The service uses `AgenticCondom` instead."""
        if not self.enabled or not text.strip():
            return text
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        return self.complete(messages).strip()
