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
from typing import Dict

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
    chunk_size:        int             = 100
    assemblyai_prompt: str             = ""
    terminology:       Dict[str, str]  = field(default_factory=dict)
    media_host:        str             = "192.168.0.113"
    media_user:        str             = "admin"

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

    return ShowConfig(
        name          = raw['name'],
        media_dir     = raw['media_dir'],
        source_lang   = raw['source_lang'],
        target_lang   = raw['target_lang'],
        system_prompt = raw['system_prompt'],
        chunk_size    = int(raw.get('chunk_size', 100)),
        assemblyai_prompt = raw.get('assemblyai_prompt', ''),
        terminology   = raw.get('terminology') or {},
        media_host    = raw.get('media_host', '192.168.0.113'),
        media_user    = raw.get('media_user', 'admin'),
        show_slug     = slug,
        state_dir     = state_dir,
        translate_log  = f"{log_base}-translate.log",
        transcribe_log = f"{log_base}-transcribe.log",
    )
