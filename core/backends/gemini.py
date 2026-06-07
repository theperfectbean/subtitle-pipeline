#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/backends/gemini.py
Translation backend using the google-genai SDK (Gemini).
"""

from typing import Tuple

import google.genai as genai
from google.genai import types
from google.api_core import exceptions as gexc

from .base import TranslationBackend, TranslationUsage, RateLimitError, ContextLengthError, TransientAPIError

# ── Pricing (per 1M tokens) ───────────────────────────────────────────────────

_PRICE_PER_M_INPUT  = 0.075
_PRICE_PER_M_OUTPUT = 0.30
_PRICE_PER_M_CACHED = 0.01875


# ── Backend ───────────────────────────────────────────────────────────────────

class GeminiTranslationBackend(TranslationBackend):
    """Gemini translation backend via google-genai SDK."""

    def __init__(self, api_key: str, system_prompt: str,
                 model: str = "gemini-3.5-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model  = model
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    @property
    def max_tokens_per_chunk(self) -> int:
        return 8192

    async def translate(self, prompt: str) -> Tuple[str, TranslationUsage]:
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config,
            )
        except gexc.ResourceExhausted as exc:
            raise RateLimitError(str(exc)) from exc
        except gexc.InvalidArgument as exc:
            msg = str(exc).lower()
            if "token" in msg or "context" in msg or "length" in msg:
                raise ContextLengthError(str(exc)) from exc
            raise TransientAPIError(str(exc)) from exc
        except Exception as exc:
            raise TransientAPIError(str(exc)) from exc

        text = response.text or ""
        usage = self._extract_usage(response)
        return text, usage

    def _extract_usage(self, response) -> TranslationUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return TranslationUsage()

        p_tokens  = getattr(meta, "prompt_token_count",          0) or 0
        ca_tokens = getattr(meta, "cached_content_token_count",  0) or 0
        can_tokens= getattr(meta, "candidates_token_count",      0) or 0
        th_tokens = getattr(meta, "thoughts_token_count",        0) or 0
        t_tokens  = getattr(meta, "total_token_count",           0) or 0

        non_cached   = max(0, p_tokens - ca_tokens)
        input_cost   = (non_cached   / 1_000_000) * _PRICE_PER_M_INPUT
        cached_cost  = (ca_tokens    / 1_000_000) * _PRICE_PER_M_CACHED
        output_cost  = ((can_tokens + th_tokens) / 1_000_000) * _PRICE_PER_M_OUTPUT
        cost         = input_cost + cached_cost + output_cost

        return TranslationUsage(
            prompt_tokens   = p_tokens,
            cached_tokens   = ca_tokens,
            output_tokens   = can_tokens,
            thoughts_tokens = th_tokens,
            total_tokens    = t_tokens,
            cost            = cost,
        )
