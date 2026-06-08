#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/transfer.py
SSH/SCP helpers and retry logic for remote media-server file transfers.
"""

import os
import shlex
import subprocess
import logging
import time

# ── Retry Helper ──────────────────────────────────────────────────────────────

def with_retry(label: str, fn, attempts: int = 3, backoff: int = 10):
    """Call fn(), retrying up to `attempts` times on any exception."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                logging.error("%s failed after %d attempts: %s", label, attempts, exc)
                raise
            logging.warning("%s failed (attempt %d/%d): %s — retrying in %ds",
                            label, attempt, attempts, exc, backoff)
            time.sleep(backoff)

# ── SSH / SCP Helpers ─────────────────────────────────────────────────────────

def run_ssh(cmd: str, host: str, user: str, check: bool = True) -> subprocess.CompletedProcess:
    """Executes a command on the media server via SSH."""
    full_cmd = [
        "timeout", "60", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        f"{user}@{host}", cmd,
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True, check=check)


def download_file(remote_path: str, local_path: str, host: str, user: str,
                  retries: int = 3, backoff: int = 10) -> None:
    """SCP a file from the media server to a local path, with retries."""
    def _do():
        subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"{user}@{host}:{remote_path}", local_path],
            check=True,
        )
    with_retry(f"download {os.path.basename(remote_path)}", _do, attempts=retries, backoff=backoff)


def upload_file(local_path: str, remote_path: str, host: str, user: str,
                retries: int = 3, backoff: int = 10) -> None:
    """Upload via remote /tmp + install so target path ownership doesn't matter."""
    remote_tmp = f"/tmp/{os.path.basename(remote_path)}.codex-upload"

    def _do():
        subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             local_path, f"{user}@{host}:{remote_tmp}"],
            check=True,
        )
        remote_cmd = (
            f"install -m 0644 {shlex.quote(remote_tmp)} {shlex.quote(remote_path)}"
            f" && rm -f {shlex.quote(remote_tmp)}"
        )
        run_ssh(remote_cmd, host, user)
    with_retry(f"upload {os.path.basename(remote_path)}", _do, attempts=retries, backoff=backoff)
