#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/backends/base.py
Abstract base classes and shared types for translation and transcription backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Quota or rate-limit hit — pipeline aborts the episode."""

class ContextLengthError(Exception):
    """Prompt exceeds model context window — pipeline splits immediately."""

class TransientAPIError(Exception):
    """Temporary network or server error — pipeline retries with backoff."""

class TranscriptionError(Exception):
    """Unrecoverable transcription failure."""


# ── Usage Dataclass ───────────────────────────────────────────────────────────

@dataclass
class TranslationUsage:
    """Token counts and estimated cost for a single translate() call."""
    prompt_tokens:   int   = 0
    cached_tokens:   int   = 0
    output_tokens:   int   = 0
    thoughts_tokens: int   = 0
    total_tokens:    int   = 0
    cost:            float = 0.0


# ── Abstract Base Classes ─────────────────────────────────────────────────────

class TranslationBackend(ABC):
    """Interface for a text-translation backend."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identifier, e.g. gemini/openai/anthropic."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Concrete model name for logging and bakeoff reporting."""

    @property
    @abstractmethod
    def max_tokens_per_chunk(self) -> int:
        """Approximate token budget per chunk for this backend/model."""

    @property
    def pricing_basis(self) -> Optional[dict]:
        """Optional price metadata for reporting."""
        return None

    @abstractmethod
    async def translate(self, prompt: str) -> Tuple[str, TranslationUsage]:
        """Submit a translation prompt, return (raw_text, usage).

        Raises:
            RateLimitError:     quota or 429 — pipeline aborts episode.
            ContextLengthError: prompt too long — pipeline splits immediately.
            TransientAPIError:  temporary error — pipeline retries with backoff.
        """


class TranscriptionBackend(ABC):
    """Interface for an audio-transcription backend."""

    @abstractmethod
    def transcribe(self, wav_path: str, language_code: str,
                   prompt: Optional[str]) -> str:
        """Submit a WAV file, return SRT text.

        Raises:
            TranscriptionError: unrecoverable failure.
        """
