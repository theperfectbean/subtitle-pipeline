#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/transfer.py
SSH/SCP helpers and retry logic for VM 113 file transfers.
"""

import os
import subprocess
import logging
import time

# ── Constants ─────────────────────────────────────────────────────────────────

VM113    = "192.168.0.113"
SSH_USER = "admin"

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

def run_ssh(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Executes a command on VM 113 via SSH."""
    full_cmd = [
        "timeout", "60", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        f"{SSH_USER}@{VM113}", cmd,
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True, check=check)


def download_file(remote_path: str, local_path: str, retries: int = 3, backoff: int = 10) -> None:
    """SCP a file from VM 113 to a local path, with retries."""
    def _do():
        subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"{SSH_USER}@{VM113}:{remote_path}", local_path],
            check=True,
        )
    with_retry(f"download {os.path.basename(remote_path)}", _do, attempts=retries, backoff=backoff)


def upload_file(local_path: str, remote_path: str, retries: int = 3, backoff: int = 10) -> None:
    """SCP a local file to VM 113, with retries."""
    def _do():
        subprocess.run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             local_path, f"{SSH_USER}@{VM113}:{remote_path}"],
            check=True,
        )
    with_retry(f"upload {os.path.basename(local_path)}", _do, attempts=retries, backoff=backoff)
