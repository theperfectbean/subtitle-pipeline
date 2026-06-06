#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate.py
CLI entry point for the subtitle translation pipeline.

Usage:
  python3 translate.py --show shows/pumuckl-1982.yaml S02E01
  python3 translate.py --show shows/pumuckl-1982.yaml S02
  python3 translate.py --show shows/pumuckl-1982.yaml all
  python3 translate.py --show shows/pumuckl-1982.yaml --dry-run S02E01
"""

import sys
import os
import re
import logging
import asyncio
import argparse

from core.config import load_show
from core.transfer import run_ssh
from core.translator import process_episode, log_usage_summary, make_usage_tracker

# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> None:
    """Configure dual stderr + file logging."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding='utf-8'),
        ],
    )

# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subtitle translation pipeline")
    p.add_argument('--show',       required=True,   help='Path to show YAML config')
    p.add_argument('--dry-run',    action='store_true', help='List matching SRT files and exit without translating')
    p.add_argument('--force',      action='store_true', help='Re-translate even if target SRT already exists')
    p.add_argument('--chunk-size', type=int, default=None, help='Override chunk_size from YAML')
    p.add_argument('target',       help='"all", "SxxExx" (e.g. S02E01), or "Sxx" (e.g. S02)')
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    # Load config before logging is set up (config derives the log path)
    cfg = load_show(args.show)
    setup_logging(cfg.translate_log)

    # CLI flags take precedence; env vars as fallback
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    dry_run = args.dry_run or os.environ.get('DRY_RUN', '0') == '1'
    force   = args.force   or os.environ.get('FORCE',   '0') == '1'
    target  = args.target

    logging.info(
        "=== Translation Run: show=%s, target=%s, DRY_RUN=%s, FORCE=%s, CHUNK_SIZE=%d ===",
        cfg.name, target, dry_run, force, cfg.chunk_size,
    )

    # Verify SSH connection to media server
    try:
        res = run_ssh("true", cfg.media_host, cfg.media_user, check=False)
        if res.returncode != 0:
            logging.critical("Unable to connect to %s via SSH. Please ensure authorization is correct.", cfg.media_host)
            sys.exit(1)
    except Exception as e:
        logging.critical("SSH to %s failed: %s", cfg.media_host, e)
        sys.exit(1)

    # Find source SRT files on media server
    try:
        res = run_ssh(f"find '{cfg.media_dir}' -name '*.{cfg.source_lang}*.srt' | sort",
                      cfg.media_host, cfg.media_user)
        srt_paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
    except Exception as e:
        logging.critical("Failed to search files on %s: %s", cfg.media_host, e)
        sys.exit(1)

    # Accumulate usage across all episodes in this run
    usage = make_usage_tracker()
    found_episodes = 0

    for srt_path in srt_paths:
        ep_match = re.search(r'S\d{2}E\d{2}', srt_path)
        if not ep_match:
            continue
        ep_id = ep_match.group(0)

        # Target matching: "all" = everything, "SxxExx" = exact episode, "Sxx" = whole season
        if target != 'all':
            if re.fullmatch(r'S\d{2}E\d{2}', target):
                if ep_id != target:
                    continue
            elif re.fullmatch(r'S\d{2}', target):
                if not ep_id.startswith(target):
                    continue
            else:
                logging.warning("Unrecognized target format: %s", target)
                sys.exit(1)

        found_episodes += 1

        # Dry-run: list matching SRTs and exit without translating
        if dry_run:
            logging.info("DRY RUN match: %s (%s)", srt_path, ep_id)
            continue

        # Locate MKV path alongside SRT
        try:
            res_mkv = run_ssh(f"find '{cfg.media_dir}' -name '*{ep_id}*.mkv' 2>/dev/null | head -1",
                              cfg.media_host, cfg.media_user, check=False)
            mkv_path = res_mkv.stdout.strip()
        except Exception:
            mkv_path = ""

        try:
            await process_episode(ep_id, srt_path, mkv_path, cfg, dry_run=False, force=force, usage=usage)
        except Exception as e:
            logging.exception("Exception raised while processing %s: %s", ep_id, e)

    if found_episodes == 0:
        logging.warning("No matching episodes found (target=%s)", target)
    elif dry_run:
        logging.info("DRY RUN complete — %d matching file(s) found, no translation performed.", found_episodes)
    else:
        log_usage_summary(usage)

    logging.info("=== Run complete ===")


if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
