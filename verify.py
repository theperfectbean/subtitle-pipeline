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
from typing import List, Optional, Tuple

from core.config import load_show, ShowConfig
from core.transfer import run_ssh, download_file
from core.srt import parse_srt

try:
    import google.genai as genai
    from google.genai import types
except ImportError:
    genai = None


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
HTML_TAG_RE = re.compile(r'</?[a-zA-Z][^>\n]{0,200}>')
HTML_MARKUP_PATTERNS = ['<br', '<font', '</', '&nbsp;']
SUSPICIOUS_LITERAL_PATTERNS = [
    (re.compile(r'\bstews?\b', re.IGNORECASE), "suspicious literal term 'stews'"),
    (re.compile(r'\bhooking hard\b', re.IGNORECASE), "suspicious literal phrase 'hooking hard'"),
    (
        re.compile(
            r'\binvest\b(?=.*\b(glasses|spectacles|red hair|see|look|wear|put|sea|seas)\b)',
            re.IGNORECASE | re.DOTALL,
        ),
        "suspicious literal term 'invest' in broken context",
    ),
]
FILENAME_TYPO_PATTERNS = [
    (re.compile(r'Kobolds law', re.IGNORECASE), "known English metadata typo: 'Kobolds law'"),
    (re.compile(r'hickup', re.IGNORECASE), "known English metadata typo: 'hickup'"),
    (re.compile(r'New Years Eve', re.IGNORECASE), "known English metadata typo: 'New Years Eve'"),
]
SEQUENCE_RE = re.compile(r'^\d+$')

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


def _normalize_match_text(text: str) -> str:
    """Collapse whitespace so term matching survives subtitle line breaks."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def _suspicious_literal_issue(text: str) -> Optional[str]:
    """Return a suspicious-literal issue for obviously broken English cues."""
    for pattern, message in SUSPICIOUS_LITERAL_PATTERNS:
        if pattern.search(text):
            return message

    normalized = _normalize_match_text(text)
    if 'red hair' in normalized and re.search(r'\b(man|guy|fellow|chap|one|with)\b', normalized):
        return "suspicious literal phrase 'red hair'"

    return None


def check_filename_metadata(filename: str) -> List[str]:
    """Report known filename/title metadata typos without changing the file."""
    issues: List[str] = []
    for pattern, message in FILENAME_TYPO_PATTERNS:
        if pattern.search(filename):
            issues.append(f"Filename/title lint: {message}")
    return issues


def check_source_warnings(blocks: List[dict], cfg: ShowConfig) -> List[str]:
    """Scan source cues for show-specific transcript quality warnings."""
    issues: List[str] = []
    for warning in cfg.source_warnings:
        pattern_text = warning.get('pattern', '')
        message = warning.get('message', '')
        if not pattern_text or not message:
            continue
        try:
            pattern = re.compile(pattern_text, re.IGNORECASE)
        except re.error as exc:
            issues.append(f"Invalid source warning pattern {pattern_text!r}: {exc}")
            continue
        for block in blocks:
            if pattern.search(block['text']):
                issues.append(f"Block {block['seq']}: source warning: {message}")
    return issues


def _normalize_srt_content(content: str) -> str:
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return content.lstrip('\ufeff')


def _raw_srt_issues(content: str, parsed_blocks: List[dict]) -> List[str]:
    """Validate the raw SRT shape before tolerant parsing smooths over defects."""
    issues: List[str] = []
    normalized = _normalize_srt_content(content).strip('\n')
    if not normalized:
        return ["SRT content is empty"]

    raw_blocks = [block for block in re.split(r'\n{2,}', normalized) if block.strip()]
    if len(raw_blocks) != len(parsed_blocks):
        issues.append(
            f"Raw block structure mismatch: found {len(raw_blocks)} raw blocks, parsed {len(parsed_blocks)} blocks"
        )

    for index, raw_block in enumerate(raw_blocks, 1):
        lines = raw_block.split('\n')
        if len(lines) < 3:
            issues.append(f"Raw block {index}: incomplete block structure")
            continue
        if not SEQUENCE_RE.match(lines[0].strip()):
            issues.append(f"Raw block {index}: invalid sequence line {lines[0]!r}")
        if '-->' not in lines[1]:
            issues.append(f"Raw block {index}: invalid timestamp line {lines[1]!r}")
        for extra_line in lines[2:]:
            stripped = extra_line.strip()
            if SEQUENCE_RE.match(stripped):
                issues.append(f"Raw block {index}: embedded sequence line {extra_line!r}")
            elif '-->' in stripped:
                issues.append(f"Raw block {index}: embedded timestamp line {extra_line!r}")

    return issues


# ── Check function ────────────────────────────────────────────────────────────

def check_srt(content: str, lang: str, cfg: Optional[ShowConfig] = None) -> List[str]:
    """Run all structural and content checks. Returns list of issue strings."""
    issues: List[str] = []

    normalized_content = _normalize_srt_content(content)
    blocks = parse_srt(normalized_content)
    issues.extend(_raw_srt_issues(normalized_content, blocks))

    # Block count
    if len(blocks) <= MIN_BLOCK_COUNT:
        issues.append(f"Block count too low: {len(blocks)} blocks")

    # Sequence checks
    seen: set = set()
    expected_seq = 1
    for b in blocks:
        if b['seq'] in seen:
            issues.append(f"Block {b['seq']}: duplicate block number")
        if not SEQUENCE_RE.match(b['seq']):
            issues.append(f"Block {b['seq']}: non-numeric block number")
        elif int(b['seq']) != expected_seq:
            issues.append(f"Block {b['seq']}: expected sequence number {expected_seq}")
        seen.add(b['seq'])
        expected_seq += 1

    # Per-block timestamp checks
    prev_start_ms = -1
    prev_end_ms = -1
    for b in blocks:
        m = TS_RE.match(b['ts'])
        if not m:
            issues.append(f"Block {b['seq']}: unparseable timestamp: {b['ts']!r}")
            continue

        start_ms = _ts_to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end_ms   = _ts_to_ms(m.group(5), m.group(6), m.group(7), m.group(8))

        if end_ms <= start_ms:
            issues.append(f"Block {b['seq']}: non-positive duration timestamp range")

        duration_s = (end_ms - start_ms) / 1000
        if duration_s > MAX_BLOCK_DURATION_S:
            issues.append(f"Block {b['seq']}: duration {duration_s:.0f}s (anomalous)")

        if prev_start_ms >= 0 and start_ms < prev_start_ms:
            issues.append(
                f"Block {b['seq']}: timestamp out of order "
                f"({_ms_to_hms(start_ms)} after {_ms_to_hms(prev_start_ms)})"
            )

        if prev_end_ms >= 0 and start_ms < prev_end_ms:
            issues.append(
                f"Block {b['seq']}: overlaps previous subtitle "
                f"({_ms_to_hms(start_ms)} starts before {_ms_to_hms(prev_end_ms)} ends)"
            )

        prev_start_ms = start_ms
        prev_end_ms = end_ms

    # Full-text content checks
    full_text  = '\n'.join(b['text'] for b in blocks)
    lower_text = full_text.lower()

    if '\ufffd' in normalized_content:
        issues.append("Unicode replacement character detected (possible decoding issue)")

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
        if cfg and lang == cfg.source_lang:
            issues.extend(check_source_warnings(blocks, cfg))

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

            # Check for untranslated stage directions
            for sd in ['(lacht)', '(weint)', '(schreit)', '(titelmusik)', '(stöhnt)']:
                if sd in padded:
                    issues.append(f"Block {b['seq']}: untranslated stage direction {sd!r}")

        for tag in REASONING_TAGS:
            if tag.lower() in lower_text:
                issues.append(f"Reasoning leakage detected: {tag!r}")

        if '**' in full_text:
            issues.append("Bold markers (**) detected")

        for marker in HTML_MARKUP_PATTERNS:
            if marker in lower_text:
                issues.append(f"HTML/subtitle markup detected in English output: {marker!r}")
                break
        else:
            if HTML_TAG_RE.search(full_text):
                issues.append("HTML-like tag detected in English output")

        for b in blocks:
            for pattern, message in SUSPICIOUS_LITERAL_PATTERNS:
                if pattern.search(b['text']):
                    issues.append(f"Block {b['seq']}: {message}")
                    break

    return issues


def check_translations(de_blocks: List[dict], en_blocks: List[dict], cfg: ShowConfig) -> List[str]:
    """Run cross-file side-by-side checks comparing German source to English target."""
    issues: List[str] = []

    if len(de_blocks) != len(en_blocks):
        issues.append(f"Block count mismatch: German has {len(de_blocks)} blocks, English has {len(en_blocks)} blocks")

    for i, (de_b, en_b) in enumerate(zip(de_blocks, en_blocks)):
        seq = de_b['seq']
        de_text = de_b['text']
        en_text = en_b['text']
        de_text_normalized = _normalize_match_text(de_text)

        # 1. Music symbol preservation check
        if '♪' in de_text and '♪' not in en_text:
            issues.append(f"Block {seq}: Music symbol ♪ is in German source but missing in English translation")

        # 2. Terminology consistency check (case-insensitive with a ±1 block window)
        failed_targets = set()
        for de_term, en_term in cfg.terminology.items():
            de_clean = de_term.lower()

            # Word boundary regex for German to avoid matching substrings (like Schreinerei for Schreiner)
            pattern_de = r""
            if de_clean[0].isalnum():
                pattern_de += r"\b"
            pattern_de += re.escape(de_clean)
            if de_clean[-1].isalnum():
                pattern_de += r"\b"

            if re.search(pattern_de, de_text.lower()) or de_clean in de_text_normalized:
                # Convert en_term to list of strings (synonyms)
                if isinstance(en_term, list):
                    en_options = [str(x).lower() for x in en_term]
                else:
                    en_options = [str(en_term).lower()]

                # Check current, previous, and next blocks in English to allow natural block-boundary splits
                found_en = False
                for en_opt in en_options:
                    # Word boundary regex for English option to ensure robust word matching
                    pattern_en = r""
                    if en_opt[0].isalnum():
                        pattern_en += r"\b"
                    pattern_en += re.escape(en_opt)
                    if en_opt[-1].isalnum():
                        pattern_en += r"\b"

                    for offset in [-1, 0, 1]:
                        idx = i + offset
                        if 0 <= idx < len(en_blocks):
                            en_candidate = en_blocks[idx]['text'].lower()
                            en_candidate_normalized = _normalize_match_text(en_blocks[idx]['text'])
                            if re.search(pattern_en, en_candidate) or en_opt in en_candidate_normalized:
                                found_en = True
                                break
                    if found_en:
                        break

                if not found_en:
                    repr_options = ", ".join(f"'{opt}'" for opt in en_options)
                    if repr_options not in failed_targets:
                        issues.append(f"Block {seq}: German term {de_term!r} found, but English translation matching {repr_options} is missing")
                        failed_targets.add(repr_options)

    return issues



def _llm_sample_indices(de_blocks: List[dict], en_blocks: List[dict], cfg: ShowConfig) -> List[int]:
    """Pick broad coverage plus blocks that look likely to contain quality issues."""
    if not de_blocks or not en_blocks:
        return []

    max_len = min(len(de_blocks), len(en_blocks))
    step = max(1, max_len // 8)
    indices = list(range(0, max_len, step))[:8]

    suspicious: List[int] = []
    for idx in range(max_len):
        en_text = en_blocks[idx]['text']
        en_lower = en_text.lower()
        de_text = de_blocks[idx]['text']

        if '**' in en_text or HTML_TAG_RE.search(en_text) or any(marker in en_lower for marker in HTML_MARKUP_PATTERNS):
            suspicious.append(idx)
            continue

        padded = f" {en_lower} "
        if sum(1 for w in DE_INDICATORS if w in padded) >= 2:
            suspicious.append(idx)
            continue

        if any(pattern.search(en_text) for pattern, _ in SUSPICIOUS_LITERAL_PATTERNS):
            suspicious.append(idx)
            continue

        for warning in cfg.source_warnings:
            pattern_text = warning.get('pattern', '')
            if not pattern_text:
                continue
            try:
                if re.search(pattern_text, de_text, re.IGNORECASE):
                    suspicious.append(idx)
                    break
            except re.error:
                continue

    for idx in suspicious:
        if idx not in indices:
            indices.append(idx)
        if len(indices) >= 12:
            break

    return indices


def evaluate_with_llm(de_blocks: List[dict], en_blocks: List[dict], client, cfg: ShowConfig) -> Tuple[str, str]:
    """Sample blocks and evaluate stylistic quality using the configured judge model."""
    if not de_blocks or not en_blocks:
        return "SKIP", "No blocks to evaluate."

    sample_indices = _llm_sample_indices(de_blocks, en_blocks, cfg)

    prompt = "Evaluate the following German to English subtitle translations for stylistic quality.\n\n"
    prompt += f"The source is {cfg.name!r}. Evaluate whether the English reads naturally for the show's context and preserves character voice, wordplay, rhymes, and culturally specific terms where relevant.\n"
    prompt += "Provide a rating of PASS or WARN on the first line, followed by a brief 1-2 sentence explanation focusing on translation quality, tone, and obvious mistranslations.\n\n"

    for idx in sample_indices:
        if idx < len(en_blocks):
            prompt += f"Block {de_blocks[idx]['seq']}:\nDE: {de_blocks[idx]['text']}\nEN: {en_blocks[idx]['text']}\n\n"
            
    try:
        response = client.models.generate_content(
            model=cfg.llm_judge_model or cfg.translation_model or cfg.gemini_model,
            contents=prompt,
        )
        text = response.text.strip()
        lines = text.split('\n')
        verdict = lines[0].strip()
        if 'WARN' in verdict.upper():
            verdict = 'WARN'
        else:
            verdict = 'PASS'
        reasoning = ' '.join(lines[1:]).strip()
        return verdict, reasoning
    except Exception as e:
        return "ERROR", f"LLM API failed: {e}"


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
    p.add_argument('--llm-judge', action='store_true', help='Run LLM-as-a-judge on a sample of blocks')
    p.add_argument(
        'target', nargs='?', default='all',
        help='"all", "SxxExx" (e.g. S01E11), or "Sxx" (e.g. S01)',
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_show(args.show)

    llm_client = None
    if getattr(args, 'llm_judge', False):
        judge_backend = (cfg.llm_judge_backend or cfg.translation_backend or "gemini").lower()
        if judge_backend != "gemini":
            logging.critical(
                "Unsupported llm_judge_backend %r. Only 'gemini' is implemented currently.",
                judge_backend,
            )
            sys.exit(1)
        if not genai:
            logging.critical("google-genai not installed. Cannot use --llm-judge.")
            sys.exit(1)
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logging.critical("GEMINI_API_KEY not found. Cannot use --llm-judge.")
            sys.exit(1)
        llm_client = genai.Client(api_key=api_key)


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

            # Local paths
            de_local = os.path.join(tmpdir, f"{ep_id}.{cfg.source_lang}.srt")
            en_local = os.path.join(tmpdir, f"{ep_id}.{cfg.target_lang}.srt")

            de_issues: List[str] = []
            en_issues: List[str] = []
            en_warnings: List[str] = check_filename_metadata(os.path.basename(en_path))

            # 1. Download and parse English target file first
            en_exists = run_ssh(
                f'test -f "{en_path}" && echo EXISTS || echo MISSING',
                cfg.media_host, cfg.media_user, check=False,
            )
            en_blocks = []
            if 'EXISTS' not in en_exists.stdout:
                en_issues.append("File missing")
                logging.warning("%s [%s]: file not found on VM 113: %s", ep_id, cfg.target_lang, en_path)
            else:
                try:
                    download_file(en_path, en_local, cfg.media_host, cfg.media_user, retries=2, backoff=5)
                    with open(en_local, 'r', encoding='utf-8', errors='replace') as f:
                        en_content = f.read()
                    if not en_content.strip():
                        en_issues.append("File is empty")
                    else:
                        en_blocks = parse_srt(en_content)
                        en_issues.extend(check_srt(en_content, cfg.target_lang, cfg))
                except Exception as e:
                    en_issues.append(f"Download or read failed: {e}")
                    logging.error("%s [%s]: download or read failed: %s", ep_id, cfg.target_lang, e)

            # 2. Determine potential German source candidates and choose the closest block count
            de_candidates = [srt_path]
            for ext in [f'.{cfg.source_lang}.hi.srt', f'.{cfg.source_lang}.hi.synced.srt']:
                cand = srt_path.replace(f'.{cfg.source_lang}.srt', ext)
                if cand not in de_candidates:
                    de_candidates.append(cand)

            best_de_path = srt_path
            best_cand_content = ""

            if en_blocks:
                best_diff = None
                for remote_cand in de_candidates:
                    exists = run_ssh(
                        f'test -f "{remote_cand}" && echo EXISTS || echo MISSING',
                        cfg.media_host, cfg.media_user, check=False,
                    )
                    if 'EXISTS' not in exists.stdout:
                        continue

                    cand_local = os.path.join(tmpdir, "temp_cand_de.srt")
                    try:
                        download_file(remote_cand, cand_local, cfg.media_host, cfg.media_user, retries=2, backoff=5)
                        with open(cand_local, 'r', encoding='utf-8', errors='replace') as f:
                            cand_content = f.read()
                        cand_blocks = parse_srt(cand_content)
                        diff = abs(len(cand_blocks) - len(en_blocks))
                        if best_diff is None or diff < best_diff:
                            best_diff = diff
                            best_de_path = remote_cand
                            best_cand_content = cand_content
                        if os.path.exists(cand_local):
                            os.remove(cand_local)
                    except Exception:
                        if os.path.exists(cand_local):
                            try:
                                os.remove(cand_local)
                            except OSError:
                                pass
                        continue

            if not best_cand_content:
                # Fallback to default srt_path if English was missing or candidates download failed
                best_de_path = srt_path
                try:
                    download_file(srt_path, de_local, cfg.media_host, cfg.media_user, retries=2, backoff=5)
                    with open(de_local, 'r', encoding='utf-8', errors='replace') as f:
                        best_cand_content = f.read()
                except Exception as e:
                    de_issues.append(f"Download failed: {e}")
                    logging.error("%s [%s]: download failed: %s", ep_id, cfg.source_lang, e)

            if best_cand_content:
                with open(de_local, 'w', encoding='utf-8') as f:
                    f.write(best_cand_content)
                if not best_cand_content.strip():
                    de_issues.append("File is empty")
                else:
                    de_issues.extend(check_srt(best_cand_content, cfg.source_lang, cfg))

                if best_de_path != srt_path:
                    logging.info("%s: Matched against better German candidate source: %s", ep_id, os.path.basename(best_de_path))

            llm_verdict = None
            llm_reasoning = None

            # Run cross-file checks whenever both files exist and parse, so one
            # issue category does not hide another in the report.
            if best_cand_content and en_blocks:
                try:
                    with open(de_local, 'r', encoding='utf-8', errors='replace') as f:
                        de_content = f.read()
                    with open(en_local, 'r', encoding='utf-8', errors='replace') as f:
                        en_content = f.read()

                    de_blocks = parse_srt(de_content)
                    en_blocks = parse_srt(en_content)

                    if de_blocks and en_blocks:
                        trans_issues = check_translations(de_blocks, en_blocks, cfg)
                        en_issues.extend(trans_issues)

                        if llm_client:
                            llm_verdict, llm_reasoning = evaluate_with_llm(de_blocks, en_blocks, llm_client, cfg)
                        
                except Exception as e:
                    en_issues.append(f"Cross-translation verification failed: {e}")

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
            if en_warnings:
                for issue in en_warnings:
                    print(f"  [{cfg.target_lang}][warn] {issue}")

            if llm_client and llm_verdict is not None:
                print(f"  [LLM Judge] {llm_verdict}: {llm_reasoning}")

            print()

        # Summary
        print(f"Summary: {total} checked, {passing} passing, {failing} failing")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    logging.info("=== Verify complete: %d checked, %d passing, %d failing ===",
                 total, passing, failing)


if __name__ == '__main__':
    main()
