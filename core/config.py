#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/config.py
YAML show config loader — defines ShowConfig and load_show().
"""

import os
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import yaml

# ── ShowConfig ─────────────────────────────────────────────────────────────────

@dataclass
class ShowConfig:
    """All configuration for a single show, loaded from a YAML file."""
    # Required fields from YAML
    name:          str
    media_dir:     str
    source_lang:   str
    target_lang:   str
    system_prompt: str

    # Optional YAML fields with defaults
    chunk_size:             int             = 100
    assemblyai_prompt:      str             = ""
    terminology:            Dict[str, str]  = field(default_factory=dict)
    media_host:             str             = "192.168.0.113"
    media_user:             str             = "admin"
    translation_backend:    str             = "gemini"
    translation_model:      str             = "gemini-3.5-flash"
    gemini_thinking_level:  str             = ""
    escalation_backend:     str             = ""
    escalation_model:       str             = ""
    escalation_thinking_level: str          = ""
    transcription_backend:  str             = "assemblyai"
    gemini_model:           str             = "gemini-3.5-flash"
    llm_judge_backend:      str             = ""
    llm_judge_model:        str             = ""
    source_warnings:        List[Dict[str, str]] = field(default_factory=list)

    # Derived at load time — not present in YAML
    show_slug:      str = ""
    state_dir:      str = ""
    translate_log:  str = ""
    transcribe_log: str = ""


# ── Slug Derivation ────────────────────────────────────────────────────────────

def _derive_slug(name: str) -> str:
    """Derive a filesystem-safe slug from a show name.

    'Pumuckl (1982)'              → 'pumuckl-1982'
    'Neue Geschichten vom Pumuckl' → 'neue-geschichten-vom-pumuckl'
    """
    s = re.sub(r'[()]', '', name).strip()
    s = s.lower().replace(' ', '-')
    s = re.sub(r'-+', '-', s)
    return s


# ── Config Loader ──────────────────────────────────────────────────────────────

_REQUIRED = ('name', 'media_dir', 'source_lang', 'target_lang', 'system_prompt')

def load_show(path: str) -> ShowConfig:
    """Load and validate a show YAML config, returning a populated ShowConfig."""
    if not os.path.exists(path):
        logging.critical("Show config not found: %s", path)
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        logging.critical("Show config is not a YAML mapping: %s", path)
        sys.exit(1)

    missing = [k for k in _REQUIRED if not raw.get(k)]
    if missing:
        logging.critical("Show config %s is missing required fields: %s", path, ', '.join(missing))
        sys.exit(1)

    slug      = _derive_slug(raw['name'])
    state_dir = f"/home/admin/subtitle-pipeline-state/{slug}"
    log_base  = f"/home/admin/logs/subtitle-pipeline-{slug}"

    os.makedirs(state_dir, exist_ok=True)
    os.makedirs("/home/admin/logs", exist_ok=True)

    translation_backend = raw.get('translation_backend', 'gemini')
    legacy_gemini_model = raw.get('gemini_model', 'gemini-3.5-flash')
    translation_model = raw.get('translation_model', legacy_gemini_model)
    escalation_backend = raw.get('escalation_backend', translation_backend)
    escalation_model = raw.get('escalation_model', '')
    gemini_thinking_level = raw.get('gemini_thinking_level', '')
    escalation_thinking_level = raw.get('escalation_thinking_level', '')

    return ShowConfig(
        name          = raw['name'],
        media_dir     = raw['media_dir'],
        source_lang   = raw['source_lang'],
        target_lang   = raw['target_lang'],
        system_prompt = raw['system_prompt'],
        chunk_size    = int(raw.get('chunk_size', 100)),
        assemblyai_prompt = raw.get('assemblyai_prompt', ''),
        terminology   = raw.get('terminology') or {},
        media_host             = raw.get('media_host', '192.168.0.113'),
        media_user             = raw.get('media_user', 'admin'),
        translation_backend    = translation_backend,
        translation_model      = translation_model,
        gemini_thinking_level  = gemini_thinking_level,
        escalation_backend     = escalation_backend,
        escalation_model       = escalation_model,
        escalation_thinking_level = escalation_thinking_level,
        transcription_backend  = raw.get('transcription_backend', 'assemblyai'),
        gemini_model           = legacy_gemini_model,
        llm_judge_backend      = raw.get('llm_judge_backend', translation_backend),
        llm_judge_model        = raw.get('llm_judge_model', translation_model),
        source_warnings        = raw.get('source_warnings') or [],
        show_slug     = slug,
        state_dir     = state_dir,
        translate_log  = f"{log_base}-translate.log",
        transcribe_log = f"{log_base}-transcribe.log",
    )
