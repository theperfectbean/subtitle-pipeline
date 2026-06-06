#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe.py
CLI entry point for the AssemblyAI transcription pipeline.

Usage:
  python3 transcribe.py --show shows/pumuckl-1982.yaml S02E01
  python3 transcribe.py --show shows/pumuckl-1982.yaml S02
  python3 transcribe.py --show shows/pumuckl-1982.yaml all
"""

import sys
import os
import re
import logging
import argparse

from core.config import load_show
from core.transfer import run_ssh
from core.transcriber import setup_assemblyai, transcribe_episode

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
    p = argparse.ArgumentParser(description="Subtitle transcription pipeline")
    p.add_argument('--show', required=True, help='Path to show YAML config')
    p.add_argument('target', help='"all", "SxxExx" (e.g. S02E01), or "Sxx" (e.g. S02)')
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = load_show(args.show)
    setup_logging(cfg.transcribe_log)
    target = args.target

    logging.info("=== Transcription Run: show=%s, target=%s ===", cfg.name, target)

    if not setup_assemblyai():
        sys.exit(1)

    # Verify SSH connection to media server
    try:
        res = run_ssh("true", cfg.media_host, cfg.media_user, check=False)
        if res.returncode != 0:
            logging.critical("Unable to connect to %s via SSH. Please ensure authorization is correct.", cfg.media_host)
            sys.exit(1)
    except Exception as e:
        logging.critical("SSH to %s failed: %s", cfg.media_host, e)
        sys.exit(1)

    # Find MKV files on media server
    try:
        res = run_ssh(f"find '{cfg.media_dir}' -name '*.mkv' | sort",
                      cfg.media_host, cfg.media_user)
        mkv_paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
    except Exception as e:
        logging.critical("Failed to search MKV files on %s: %s", cfg.media_host, e)
        sys.exit(1)

    results      = {}
    found_episodes = 0

    for mkv_path in mkv_paths:
        ep_match = re.search(r'S\d{2}E\d{2}', mkv_path)
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
        # basename is the full filename without .mkv extension
        basename = os.path.basename(mkv_path)
        if basename.endswith('.mkv'):
            basename = basename[:-4]

        ok = transcribe_episode(basename, cfg)
        results[ep_id] = 'ok' if ok else 'FAILED'

    if found_episodes == 0:
        logging.warning("No matching episodes found (target=%s)", target)
    else:
        logging.info("=== SUMMARY ===")
        for ep_id, status in results.items():
            logging.info("  %s  %s", status, ep_id)

        failed = [ep for ep, s in results.items() if s != 'ok']
        if failed:
            logging.error("%d episode(s) failed — see log above", len(failed))
            sys.exit(1)
        logging.info("All episodes completed successfully.")

    logging.info("=== Run complete ===")


if __name__ == '__main__':
    main()
