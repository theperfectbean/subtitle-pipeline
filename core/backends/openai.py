#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/backends/openai.py
Translation backend using the OpenAI Chat Completions API over HTTPS.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

from .base import (
    ContextLengthError,
    RateLimitError,
    TransientAPIError,
    TranslationBackend,
    TranslationUsage,
)

_API_URL = "https://api.openai.com/v1/chat/completions"

_MODEL_PRICING = {
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
    "gpt-5.4-mini": {
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.50,
    },
    "gpt-5.4-nano": {
        "input": 0.20,
        "cached_input": 0.020,
        "output": 1.25,
    },
    "gpt-5.4-pro": {
        "input": 30.00,
        "cached_input": 0.0,
        "output": 180.00,
    },
}


def _pricing_for_model(model: str) -> Dict[str, float]:
    model = model.lower()
    for prefix in sorted(_MODEL_PRICING, key=len, reverse=True):
        pricing = _MODEL_PRICING[prefix]
        if model == prefix or model.startswith(f"{prefix}-"):
            return pricing
    return _MODEL_PRICING["gpt-5.4"]


class OpenAITranslationBackend(TranslationBackend):
    """OpenAI translation backend via HTTPS."""

    def __init__(self, api_key: str, system_prompt: str, model: str = "gpt-5.4") -> None:
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._model = model
        self._pricing = _pricing_for_model(model)

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens_per_chunk(self) -> int:
        return 8192

    @property
    def pricing_basis(self) -> dict:
        return dict(self._pricing)

    async def translate(self, prompt: str) -> Tuple[str, TranslationUsage]:
        response = await asyncio.to_thread(self._request, prompt)
        text = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ) or ""
        usage = self._extract_usage(response.get("usage") or {})
        return text, usage

    def _request(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise RateLimitError(detail) from exc
            if exc.code == 400 and any(term in detail.lower() for term in ("token", "context", "length")):
                raise ContextLengthError(detail) from exc
            raise TransientAPIError(detail) from exc
        except urllib.error.URLError as exc:
            raise TransientAPIError(str(exc)) from exc

    def _extract_usage(self, usage: Dict[str, Any]) -> TranslationUsage:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + output_tokens) or 0)
        cached_tokens = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0)
        non_cached = max(0, prompt_tokens - cached_tokens)
        cost = (
            (non_cached / 1_000_000) * self._pricing["input"]
            + (cached_tokens / 1_000_000) * self._pricing["cached_input"]
            + (output_tokens / 1_000_000) * self._pricing["output"]
        )
        return TranslationUsage(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
