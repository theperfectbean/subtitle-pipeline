#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/backends/assemblyai.py
Transcription backend using the AssemblyAI SDK.
"""

from typing import Optional

import assemblyai as aai

from .base import TranscriptionBackend, TranscriptionError


class AssemblyAITranscriptionBackend(TranscriptionBackend):
    """AssemblyAI Universal-3 Pro transcription backend."""

    def __init__(self, api_key: str) -> None:
        aai.settings.api_key = api_key

    def transcribe(self, wav_path: str, language_code: str,
                   prompt: Optional[str]) -> str:
        config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro"],
            language_code=language_code,
            prompt=prompt,
        )
        try:
            transcriber = aai.Transcriber()
            transcript  = transcriber.transcribe(wav_path, config=config)
            if transcript.status == aai.TranscriptStatus.error:
                raise TranscriptionError(
                    f"AssemblyAI transcript error: {transcript.error}"
                )
            return transcript.export_subtitles_srt()
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc
