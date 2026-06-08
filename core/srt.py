#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/srt.py
SRT parsing, rendering, validation, and text-wrapping — pure functions, no I/O.
"""

import re
import textwrap
from typing import Dict, List, Tuple

_TS_START_RE = re.compile(r'^(\d{2}):(\d{2}):(\d{2}),(\d{1,3})')
_FONT_TAG_RE = re.compile(r'</?font\b[^>]*>', re.IGNORECASE)
_BR_TAG_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r'</?[a-zA-Z][^>\n]{0,200}>')

# ── SRT Parsing & Normalization ───────────────────────────────────────────────

def parse_srt(content: str) -> List[Dict[str, str]]:
    """
    Parses SRT content into a list of block dicts.
    Normalizes CRLF to LF and strips UTF-8 BOM.
    """
    # Normalize CRLF and CR to LF
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    if not content.endswith('\n'):
        content += '\n'

    blocks = []
    current = []
    next_seq = 1

    def _append_block(lines: List[str]) -> None:
        nonlocal next_seq
        if len(lines) < 2:
            return

        seq = ""
        ts_index = 1
        if re.match(r'^\d+$', lines[0]) and '-->' in lines[1]:
            seq = lines[0]
        elif '-->' in lines[0]:
            seq = str(next_seq)
            ts_index = 0
        else:
            return

        text = '\n'.join(lines[ts_index + 1:]) if len(lines) > ts_index + 1 else ''
        if text.strip():
            blocks.append({'seq': seq, 'ts': lines[ts_index], 'text': text})
            next_seq += 1

    for line in content.split('\n'):
        if line == '':
            if current:
                _append_block(current)
            current = []
        else:
            current.append(line)

    if current:
        _append_block(current)

    return blocks


def render_blocks(blocks: List[Dict[str, str]], renumber: bool = True) -> str:
    """Renders a list of block dicts back into valid SRT string format."""
    parts = []
    for i, b in enumerate(blocks, 1):
        seq = str(i) if renumber else b.get('seq', str(i))
        parts.append(f"{seq}\n{b['ts']}\n{b['text']}")
    if parts:
        return '\n\n'.join(parts) + '\n\n'
    return ""


def normalize_subtitle_text(text: str) -> str:
    """Normalize subtitle text artifacts that should never be rendered visibly."""
    text = _BR_TAG_RE.sub('\n', text)
    text = _FONT_TAG_RE.sub('', text)
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'[ \t]+([,.;:!?])', r'\1', text)
    text = re.sub(r'\n(?:[ \t]*\n){2,}', '\n\n', text)
    return text


def wrap_subtitle_text(text: str, width: int = 45) -> str:
    """Programmatically wraps subtitle text to ensure readable line lengths on screen."""
    text = normalize_subtitle_text(text)
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if len(line) > width:
            wrapped = textwrap.fill(line, width=width, break_long_words=False)
            wrapped_lines.append(wrapped)
        else:
            wrapped_lines.append(line)
    return '\n'.join(wrapped_lines)


def sort_and_renumber(blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Sort blocks by start timestamp and renumber from 1.

    Applied as a final global pass after all chunks are stitched so that any
    out-of-order blocks produced by the split/retry logic are corrected before
    the file is written to disk.
    """
    def _start_ms(block: Dict[str, str]) -> int:
        m = _TS_START_RE.match(block.get('ts', ''))
        if not m:
            return 0
        h, mm, s, ms_str = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        ms = int(ms_str.ljust(3, '0'))
        return (h * 3600 + mm * 60 + s) * 1000 + ms

    sorted_blocks = sorted(blocks, key=_start_ms)
    for i, b in enumerate(sorted_blocks, 1):
        b['seq'] = str(i)
    return sorted_blocks


def strip_fences(text: str) -> str:
    """Removes markdown code block fences and leading lines before block 1."""
    lines = text.splitlines()
    out = []
    in_fence = False
    for line in lines:
        if line.strip().startswith('```'):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)

    # Find the start index where the first sequence number lies
    start = 0
    for i, l in enumerate(out):
        if re.match(r'^\d+$', l.strip()):
            start = i
            break
    return '\n'.join(out[start:])


# ── Structural Validation ─────────────────────────────────────────────────────

def validate_translation_structure(raw_text: str, chunk_blocks: List[Dict[str, str]]) -> Tuple[bool, str]:
    """
    Stage 1: Programmatic structural check on the raw worker output.
    Checks:
    - No markdown fences (```)
    - No leaked reasoning text or headers (like <thinking> or typical chain-of-thought lines)
    - Block count match
    - Count of '-->' matches expected block count (timestamps force-overwritten from source after this)
    - Count of '-->' matches exactly
    """
    expected_count = len(chunk_blocks)

    # 1. No markdown fences
    if "```" in raw_text:
        return False, "Markdown code fences (```) detected."

    # 2. No HTML/subtitle markup in model output
    lower_raw = raw_text.lower()
    html_indicators = ["<br", "</font>", "<font", "&nbsp;"]
    if any(indicator in lower_raw for indicator in html_indicators) or _HTML_TAG_RE.search(raw_text):
        return False, "HTML or subtitle markup detected; output plain SRT text only with real line breaks."

    # 3. No leaked reasoning text / headers / preamble
    # Note: ** markers are stripped before this runs, so bold-wrapped indicators
    # (e.g. "**thinking**") would never match and are intentionally omitted.
    reasoning_indicators = [
        "<thinking>", "</thinking>", "[thinking]", "(thinking)",
        "thinking:", "reasoning:", "my focus is on", "i'm now focused",
        "i am now structuring", "i need to be careful", "api glitch", "token limit",
        "(wait,", "(note:", "(correction:",
    ]
    for indicator in reasoning_indicators:
        if indicator in lower_raw:
            return False, f"Leaked reasoning or metadata text detected (found indicator '{indicator}')."

    # 4. Check for preamble / postamble:
    # First non-empty line of the raw text must be a sequence number (a digit)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return False, "Output is completely empty."

    first_line_clean = lines[0].lstrip('﻿')
    if not first_line_clean.isdigit():
        return False, f"Preamble detected (output does not start with a sequence number): '{lines[0][:50]}...'"

    # 5. Parse blocks
    parsed_blocks = parse_srt(raw_text)

    # 6. Block count match
    if len(parsed_blocks) != expected_count:
        return False, f"Block count mismatch: expected {expected_count} blocks, got {len(parsed_blocks)}."

    # 7. Timestamp preservation & occurrences check
    arrow_count = raw_text.count("-->")
    if arrow_count != expected_count:
        return False, f"Timestamp count mismatch: expected {expected_count} '-->' indicators, got {arrow_count}."

    return True, ""
