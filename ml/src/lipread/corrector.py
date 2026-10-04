"""LLM corrector: noisy lip-reading text → clean English.

Talks to any OpenAI-compatible chat endpoint so the same code works for a llama.cpp/vLLM server on
the RunPod pod (fine-tuned Llama-3.2-3B), Cloudflare Workers AI, OpenRouter, or Ollama locally.
Unconfigured → passthrough, so the pipeline never blocks on it.
"""

from __future__ import annotations

import os

import httpx

SYSTEM_PROMPT = (
    "You fix the output of a lip-reading model. The input is a noisy transcript of what a person "
    "silently mouthed; words may be wrong but usually look similar on the lips (e.g. p/b/m, f/v). "
    "Return only the most likely intended sentence in plain English, no quotes, no explanation. "
    "If the input is empty or meaningless, return it unchanged."
)


class Corrector:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 10.0):
        self.base_url = (base_url or os.environ.get("CORRECTOR_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("CORRECTOR_MODEL", "")
        self.api_key = api_key or os.environ.get("CORRECTOR_API_KEY", "")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def correct(self, text: str) -> str:
        if not self.enabled or not text.strip():
            return text
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            timeout=self.timeout,
            json={
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 128,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
