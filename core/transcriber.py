#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/transcriber.py
AssemblyAI Universal-3 Pro transcription pipeline, generalised via ShowConfig.
"""

import os
import logging
import tempfile
import subprocess
import shutil
from typing import Optional

import assemblyai as aai

from .config import ShowConfig
from .transfer import run_ssh, download_file, upload_file, with_retry

# ── API Key ───────────────────────────────────────────────────────────────────

_AAI_KEY_PATH = "/home/admin/.assemblyai_api_key"

# ── Bootstrap ─────────────────────────────────────────────────────────────────

def setup_assemblyai() -> bool:
    """Read AssemblyAI API key and configure the SDK. Returns False on failure."""
    if not os.path.exists(_AAI_KEY_PATH):
        logging.error("AssemblyAI API key not found at %s", _AAI_KEY_PATH)
        return False
    with open(_AAI_KEY_PATH, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        logging.error("AssemblyAI API key at %s is empty", _AAI_KEY_PATH)
        return False
    aai.settings.api_key = key
    return True

# ── Audio Extraction ──────────────────────────────────────────────────────────

def extract_audio(mkv_path: str, wav_path: str) -> bool:
    """Extract first audio stream from MKV as lossless PCM WAV. Returns True on success."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", mkv_path,
         "-map", "0:a:0", "-acodec", "pcm_s16le", "-ar", "48000", "-f", "wav", wav_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logging.error("ffmpeg failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
        return False
    return True

# ── AssemblyAI Transcription ──────────────────────────────────────────────────

def transcribe_audio(wav_path: str, cfg: ShowConfig) -> Optional[str]:
    """Submit WAV to AssemblyAI, wait for completion, return SRT text or None."""
    # Treat placeholder/empty prompts as no prompt
    prompt = cfg.assemblyai_prompt.strip()
    if not prompt or prompt.startswith('#'):
        prompt = None

    config = aai.TranscriptionConfig(
        speech_models=["universal-3-pro"],
        language_code=cfg.source_lang,
        prompt=prompt,
    )

    def _do():
        transcriber = aai.Transcriber()
        transcript  = transcriber.transcribe(wav_path, config=config)
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcript error: {transcript.error}")
        return transcript.export_subtitles_srt()

    try:
        return with_retry("AAI transcribe", _do)
    except Exception:
        return None

# ── Per-Episode Transcription ─────────────────────────────────────────────────

def transcribe_episode(mkv_basename: str, cfg: ShowConfig) -> bool:
    """Full transcription pipeline for one episode. Returns True if source-lang SRT was produced."""
    sl         = cfg.source_lang
    mkv_remote = f"{cfg.media_dir}/{mkv_basename}.mkv"
    srt_remote = f"{cfg.media_dir}/{mkv_basename}.{sl}.srt"

    # Skip if already done
    check = run_ssh(f'test -f "{srt_remote}" && echo EXISTS || echo MISSING', check=False)
    if check.returncode == 0 and "EXISTS" in check.stdout:
        logging.info("SKIP %s — .%s.srt already exists", mkv_basename, sl)
        return True

    logging.info("=== START: %s ===", mkv_basename)
    tmpdir    = tempfile.mkdtemp(prefix="subtitle-pipeline-aai-")
    mkv_local = os.path.join(tmpdir, f"{mkv_basename}.mkv")
    wav_local = os.path.join(tmpdir, f"{mkv_basename}.wav")
    srt_local = os.path.join(tmpdir, f"{mkv_basename}.{sl}.srt")

    try:
        # 1. Download MKV from VM 113
        logging.info("Downloading MKV from VM 113 …")
        download_file(mkv_remote, mkv_local)

        # 2. Extract audio
        logging.info("Extracting audio …")
        if not extract_audio(mkv_local, wav_local):
            logging.error("Audio extraction failed for %s — skipping", mkv_basename)
            return False

        # Free MKV disk space before AAI upload
        os.remove(mkv_local)

        # 3. Transcribe via AssemblyAI
        logging.info("Submitting to AssemblyAI Universal-3 Pro …")
        srt_text = transcribe_audio(wav_local, cfg)
        if not srt_text:
            logging.error("Transcription returned no output for %s", mkv_basename)
            return False

        # 4. Write SRT locally, upload to VM 113
        with open(srt_local, "w", encoding="utf-8") as fh:
            fh.write(srt_text)

        logging.info("Uploading .%s.srt to VM 113 …", sl)
        upload_file(srt_local, srt_remote)

        # 5. Verify
        verify = run_ssh(f'ls -la "{srt_remote}"', check=False)
        if verify.returncode != 0:
            logging.error("Verification failed — .%s.srt not found on VM 113 after upload", sl)
            return False
        logging.info("Verified: %s", verify.stdout.strip())
        logging.info("=== DONE: %s ===", mkv_basename)
        return True

    except Exception as exc:
        logging.error("Unhandled error for %s: %s", mkv_basename, exc)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
