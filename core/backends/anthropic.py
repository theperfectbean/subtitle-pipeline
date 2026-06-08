#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/backends/anthropic.py
Translation backend using the Anthropic Messages API over HTTPS.
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

_API_URL = "https://api.anthropic.com/v1/messages"
_MODEL_PRICING = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
    },
}


def _pricing_for_model(model: str) -> Dict[str, float]:
    model = model.lower()
    for prefix in sorted(_MODEL_PRICING, key=len, reverse=True):
        pricing = _MODEL_PRICING[prefix]
        if model == prefix or model.startswith(f"{prefix}-"):
            return pricing
    return _MODEL_PRICING["claude-sonnet-4-6"]


class AnthropicTranslationBackend(TranslationBackend):
    """Anthropic translation backend via HTTPS."""

    def __init__(self, api_key: str, system_prompt: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._model = model
        self._pricing = _pricing_for_model(model)

    @property
    def provider(self) -> str:
        return "anthropic"

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
        parts = []
        for block in response.get("content") or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "".join(parts)
        usage = self._extract_usage(response.get("usage") or {})
        return text, usage

    def _request(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self._model,
            "system": self._system_prompt,
            "max_tokens": 8192,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _API_URL,
            data=body,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
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
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = prompt_tokens + output_tokens
        cost = (
            (prompt_tokens / 1_000_000) * self._pricing["input"]
            + (output_tokens / 1_000_000) * self._pricing["output"]
        )
        return TranslationUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
