#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py
Quality-verification script for subtitle output.

Usage:
  python3 verify.py --show shows/pumuckl-1982.yaml
  python3 verify.py --show shows/pumuckl-1982.yaml S02
  python3 verify.py --show shows/pumuckl-1982.yaml S01E11
"""

import sys
import os
import re
import logging
import argparse
import tempfile
import shutil
from typing import List, Tuple

from core.config import load_show, ShowConfig
from core.transfer import run_ssh, download_file
from core.srt import parse_srt

# ── Constants ─────────────────────────────────────────────────────────────────

TS_RE = re.compile(
    r'^(\d{2}):(\d{2}):(\d{2}),(\d{1,3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{1,3})$'
)

MAX_BLOCK_DURATION_S = 60
MIN_BLOCK_COUNT      = 10

ENCODING_ARTEFACTS = ['Ã¼', 'Ã¶', 'Ã¤', 'â€', 'Ã']
SPAM_PATTERNS      = ['osdb.link', 'subscene', 'opensubtitles', 'addic7ed']

EN_FRAGMENTS   = [' the ', ' and ', ' of ']
DE_INDICATORS  = [' der ', ' die ', ' das ', ' und ', ' ist ', ' nicht ', ' dass ']
REASONING_TAGS = ['<thinking>', '</thinking>', '**thinking**', 'reasoning:']

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def _ms_to_hms(ms: int) -> str:
    s_total = ms // 1000
    h = s_total // 3600
    m = (s_total % 3600) // 60
    s = s_total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ep_title(srt_basename: str, source_lang: str) -> str:
    """'S01E11 - Pumuckl und der Nikolaus.de.srt' → 'S01E11 - Pumuckl und der Nikolaus'"""
    title = re.sub(r'\.' + re.escape(source_lang) + r'\.srt$', '', srt_basename, flags=re.IGNORECASE)
    return title.strip()


# ── Check function ────────────────────────────────────────────────────────────

def check_srt(content: str, lang: str) -> List[str]:
    """Run all structural and content checks. Returns list of issue strings."""
    issues: List[str] = []

    blocks = parse_srt(content)

    # Block count
    if len(blocks) <= MIN_BLOCK_COUNT:
        issues.append(f"Block count too low: {len(blocks)} blocks")

    # Duplicate sequence numbers
    seen: set = set()
    for b in blocks:
        if b['seq'] in seen:
            issues.append(f"Block {b['seq']}: duplicate block number")
        seen.add(b['seq'])

    # Per-block timestamp checks
    prev_start_ms = -1
    for b in blocks:
        m = TS_RE.match(b['ts'])
        if not m:
            issues.append(f"Block {b['seq']}: unparseable timestamp: {b['ts']!r}")
            continue

        start_ms = _ts_to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end_ms   = _ts_to_ms(m.group(5), m.group(6), m.group(7), m.group(8))

        duration_s = (end_ms - start_ms) / 1000
        if duration_s > MAX_BLOCK_DURATION_S:
            issues.append(f"Block {b['seq']}: duration {duration_s:.0f}s (anomalous)")

        if prev_start_ms >= 0 and start_ms < prev_start_ms:
            issues.append(
                f"Block {b['seq']}: timestamp out of order "
                f"({_ms_to_hms(start_ms)} after {_ms_to_hms(prev_start_ms)})"
            )

        prev_start_ms = start_ms

    # Full-text content checks
    full_text  = '\n'.join(b['text'] for b in blocks)
    lower_text = full_text.lower()

    for artefact in ENCODING_ARTEFACTS:
        if artefact in full_text:
            issues.append(f"Encoding artefact detected: {artefact!r}")
            break

    for spam in SPAM_PATTERNS:
        if spam in lower_text:
            issues.append(f"Spam watermark detected: {spam!r}")

    if '```' in full_text:
        issues.append("Markdown code fences (```) detected")

    # Language-specific checks
    if lang == 'de':
        for b in blocks:
            padded = f" {b['text'].lower()} "
            for word in EN_FRAGMENTS:
                if word in padded:
                    issues.append(
                        f"Block {b['seq']}: possible English fragment (found {word.strip()!r})"
                    )
                    break

    elif lang == 'en':
        for b in blocks:
            padded = f" {b['text'].lower()} "
            count = sum(1 for w in DE_INDICATORS if w in padded)
            if count >= 2:
                issues.append(
                    f"Block {b['seq']}: possible untranslated German ({count} indicators)"
                )

        for tag in REASONING_TAGS:
            if tag.lower() in lower_text:
                issues.append(f"Reasoning leakage detected: {tag!r}")

        if '**' in full_text:
            issues.append("Bold markers (**) detected")

    return issues


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding='utf-8'),
        ],
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subtitle quality verification")
    p.add_argument('--show', required=True, help='Path to show YAML config')
    p.add_argument(
        'target', nargs='?', default='all',
        help='"all", "SxxExx" (e.g. S01E11), or "Sxx" (e.g. S01)',
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_show(args.show)

    verify_log = (
        f"/home/admin/logs/subtitle-pipeline-{cfg.show_slug}-verify.log"
    )
    setup_logging(verify_log)

    target = args.target
    logging.info(
        "=== Verify Run: show=%s, target=%s ===", cfg.name, target
    )

    # Validate target format
    if (target != 'all'
            and not re.fullmatch(r'S\d{2}E\d{2}', target)
            and not re.fullmatch(r'S\d{2}', target)):
        logging.critical("Unrecognized target format: %s", target)
        sys.exit(1)

    # Verify SSH
    try:
        res = run_ssh("true", cfg.media_host, cfg.media_user, check=False)
        if res.returncode != 0:
            logging.critical("Unable to connect to VM 113 via SSH.")
            sys.exit(1)
    except Exception as e:
        logging.critical("SSH to VM 113 failed: %s", e)
        sys.exit(1)

    # Discover source SRT files
    try:
        res = run_ssh(
            f"find '{cfg.media_dir}' -name '*.{cfg.source_lang}.srt' | sort",
            cfg.media_host, cfg.media_user,
        )
        srt_paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
    except Exception as e:
        logging.critical("Failed to search files on VM 113: %s", e)
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="subtitle-pipeline-verify-")
    try:
        total   = 0
        passing = 0
        failing = 0
        all_issues: List[Tuple[str, str, str]] = []  # (ep_id, lang, issue)

        for srt_path in srt_paths:
            ep_match = re.search(r'S\d{2}E\d{2}', srt_path)
            if not ep_match:
                continue
            ep_id = ep_match.group(0)

            if target != 'all':
                if re.fullmatch(r'S\d{2}E\d{2}', target):
                    if ep_id != target:
                        continue
                elif re.fullmatch(r'S\d{2}', target):
                    if not ep_id.startswith(target):
                        continue

            total += 1
            basename = os.path.basename(srt_path)
            title    = _ep_title(basename, cfg.source_lang)

            # Derive target-lang path
            en_path = srt_path.replace(
                f'.{cfg.source_lang}.srt', f'.{cfg.target_lang}.srt'
            )

            # Fetch both files
            de_local = os.path.join(tmpdir, f"{ep_id}.{cfg.source_lang}.srt")
            en_local = os.path.join(tmpdir, f"{ep_id}.{cfg.target_lang}.srt")

            de_issues: List[str] = []
            en_issues: List[str] = []

            for remote, local, lang, issue_list in [
                (srt_path, de_local, cfg.source_lang, de_issues),
                (en_path,  en_local, cfg.target_lang, en_issues),
            ]:
                exists = run_ssh(
                    f'test -f "{remote}" && echo EXISTS || echo MISSING',
                    cfg.media_host, cfg.media_user, check=False,
                )
                if 'EXISTS' not in exists.stdout:
                    issue_list.append("File missing")
                    logging.warning("%s [%s]: file not found on VM 113: %s", ep_id, lang, remote)
                    continue

                try:
                    download_file(remote, local, cfg.media_host, cfg.media_user,
                                  retries=2, backoff=5)
                except Exception as e:
                    issue_list.append(f"Download failed: {e}")
                    logging.error("%s [%s]: download failed: %s", ep_id, lang, e)
                    continue

                try:
                    with open(local, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except Exception as e:
                    issue_list.append(f"Read failed: {e}")
                    continue

                if not content.strip():
                    issue_list.append("File is empty")
                    continue

                issue_list.extend(check_srt(content, lang))

            # Determine pass/fail
            ep_issues = (
                [(cfg.source_lang, i) for i in de_issues]
                + [(cfg.target_lang, i) for i in en_issues]
            )
            ep_passed = len(ep_issues) == 0

            if ep_passed:
                passing += 1
            else:
                failing += 1
                for lang, issue in ep_issues:
                    all_issues.append((ep_id, lang, issue))

            # Print episode result
            header = f"{title}"
            status = "PASS" if ep_passed else "FAIL"
            print(f"{header:<50}  {status}")

            if not ep_passed:
                for lang, issue in [(cfg.target_lang, i) for i in en_issues]:
                    print(f"  [{lang}] {issue}")
                if de_issues:
                    for issue in de_issues:
                        print(f"  [{cfg.source_lang}] {issue}")
                elif en_issues:
                    print(f"  [{cfg.source_lang}] No issues")

            print()

        # Summary
        print(f"Summary: {total} checked, {passing} passing, {failing} failing")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    logging.info("=== Verify complete: %d checked, %d passing, %d failing ===",
                 total, passing, failing)


if __name__ == '__main__':
    main()
