#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/backends/openai_stub.py
Stub: OpenAI translation backend.

To add OpenAI support:
  1. pip install openai
  2. Fill in the translate() method below.
  3. Add OPENAI_API_KEY to .env
  4. In translate.py _make_translation_backend(), add:
       elif cfg.translation_backend == "openai":
           return OpenAITranslationBackend(os.environ["OPENAI_API_KEY"],
                                           cfg.system_prompt)
  5. Set translation_backend: "openai" in your show YAML.
"""

from typing import Tuple

from .base import TranslationBackend, TranslationUsage, RateLimitError, TransientAPIError


class OpenAITranslationBackend(TranslationBackend):
    """Stub — replace with a real implementation."""

    def __init__(self, api_key: str, system_prompt: str,
                 model: str = "gpt-4o") -> None:
        self._api_key      = api_key
        self._system_prompt = system_prompt
        self._model        = model
        # import openai; self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def max_tokens_per_chunk(self) -> int:
        return 4096

    async def translate(self, prompt: str) -> Tuple[str, TranslationUsage]:
        raise NotImplementedError(
            "OpenAI backend is a stub — implement translate() before use."
        )
        # Example implementation:
        #
        # try:
        #     response = await self._client.chat.completions.create(
        #         model=self._model,
        #         messages=[
        #             {"role": "system", "content": self._system_prompt},
        #             {"role": "user",   "content": prompt},
        #         ],
        #     )
        # except openai.RateLimitError as exc:
        #     raise RateLimitError(str(exc)) from exc
        # except openai.APIError as exc:
        #     raise TransientAPIError(str(exc)) from exc
        #
        # text  = response.choices[0].message.content or ""
        # usage = TranslationUsage(
        #     prompt_tokens = response.usage.prompt_tokens,
        #     output_tokens = response.usage.completion_tokens,
        #     total_tokens  = response.usage.total_tokens,
        #     cost          = 0.0,  # fill in pricing
        # )
        # return text, usage
