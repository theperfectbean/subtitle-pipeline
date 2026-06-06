#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/translator.py
All translation machinery — Worker agent, chunking, state engine, retry/split logic.
"""

import sys
import os
import json
import logging
import asyncio
import re
import math
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from google.antigravity import Agent, LocalAgentConfig, GenerationConfig, ThinkingLevel
from google.antigravity.types import GeminiConfig, ModelConfig, ModelEntry
from google.antigravity.hooks import policy

from .config import ShowConfig
from .srt import parse_srt, render_blocks, wrap_subtitle_text, strip_fences, validate_translation_structure
from .transfer import run_ssh, download_file, upload_file

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CHUNK_SIZE  = 1
MAX_RETRIES     = 3
API_CALL_BUDGET = 200   # hard ceiling on API calls per episode before aborting
WORKER_TIMEOUT  = 120   # seconds before a single model call is considered hung

# Pricing rates for gemini-3.5-flash
PRICE_PER_M_INPUT  = 0.075
PRICE_PER_M_OUTPUT = 0.30
PRICE_PER_M_CACHED = 0.01875

# ── API Key ───────────────────────────────────────────────────────────────────

_API_KEY_PATH = "/home/admin/.google_api_key"
_API_KEY      = ""

if os.path.exists(_API_KEY_PATH):
    with open(_API_KEY_PATH, "r", encoding="utf-8") as _f:
        _API_KEY = _f.read().strip()

if _API_KEY:
    os.environ["GEMINI_API_KEY"] = _API_KEY
    os.environ["GOOGLE_API_KEY"] = _API_KEY

# ── Worker Config Factory ─────────────────────────────────────────────────────

def make_worker_config(system_prompt: str) -> LocalAgentConfig:
    """Build a LocalAgentConfig for the translation Worker from the show's system prompt."""
    gemini_cfg = GeminiConfig(
        api_key=_API_KEY if _API_KEY else None,
        models=ModelConfig(
            default=ModelEntry(
                name="gemini-3.5-flash",
                generation=GenerationConfig(thinking_level=ThinkingLevel.MINIMAL)
            )
        )
    )
    return LocalAgentConfig(
        system_instructions=system_prompt,
        policies=[policy.allow_all()],
        workspaces=[],
        gemini_config=gemini_cfg,
    )

# ── Usage Tracker ─────────────────────────────────────────────────────────────

def make_usage_tracker() -> Dict[str, Any]:
    """Return a fresh usage accumulator dict."""
    return {
        'prompt_tokens':     0,
        'cached_tokens':     0,
        'candidates_tokens': 0,
        'thoughts_tokens':   0,
        'total_tokens':      0,
        'cost':              0.0,
    }

def log_usage_summary(usage: Dict[str, Any]) -> None:
    """Log the cumulative token and cost summary for a translation run."""
    logging.info("=== Usage & Cost Summary ===")
    logging.info("Total Prompt Tokens:     %d", usage['prompt_tokens'])
    logging.info("Total Cached Tokens:     %d", usage['cached_tokens'])
    logging.info("Total Output Tokens:     %d", usage['candidates_tokens'])
    logging.info("Total Thoughts Tokens:   %d", usage['thoughts_tokens'])
    logging.info("Total Cumulative Tokens: %d", usage['total_tokens'])
    logging.info("Estimated Total Cost:    $%.6f", usage['cost'])
    logging.info("============================")

# ── State Engine ──────────────────────────────────────────────────────────────

def load_state(episode: str, state_dir: str) -> Dict[str, Any]:
    """Load episode state from JSON, returning {} if none exists."""
    state_file = os.path.join(state_dir, f"{episode}.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(episode: str, state: Dict[str, Any], state_dir: str) -> None:
    """Atomically persist episode state via tempfile + os.replace."""
    state_file = os.path.join(state_dir, f"{episode}.json")
    tmp_file   = state_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_file, state_file)


def init_state(episode: str, source_blocks: int, chunk_size: int, state_dir: str) -> Dict[str, Any]:
    """Create and persist fresh episode state with pending chunks."""
    n = math.ceil(source_blocks / chunk_size)
    chunks = []
    for i in range(n):
        s = i * chunk_size
        e = min(s + chunk_size, source_blocks)
        chunks.append({
            'id':     i,
            'start':  s,
            'end':    e,
            'status': 'pending',
            'output': os.path.join(state_dir, f"{episode}_chunk_{i}_final.srt"),
        })
    state = {
        'episode':       episode,
        'source_blocks': source_blocks,
        'chunk_size':    chunk_size,
        'chunks':        chunks,
        'status':        'in_progress',
    }
    save_state(episode, state, state_dir)
    return state


def resume_state(episode: str, state_dir: str) -> Dict[str, Any]:
    """Load state, reset any chunks whose output files have gone missing."""
    state = load_state(episode, state_dir)
    if not state:
        return {}

    for chunk in state['chunks']:
        if chunk['status'] == 'done' and not os.path.exists(chunk['output']):
            logging.warning("Cached chunk output file %s not found. Resetting to pending.", chunk['output'])
            chunk['status'] = 'pending'

    save_state(episode, state, state_dir)
    return state

# ── Pre-flight Checks ─────────────────────────────────────────────────────────

def preflight(srt_path: str, host: str, user: str) -> int:
    """Performs safety checks on source file and returns block count (0 = fail)."""
    try:
        res = run_ssh(f"grep -cE '^[[:space:]]*[0-9]+[[:space:]]*$' '{srt_path}'",
                      host, user, check=False)
        block_count = int(res.stdout.strip()) if res.stdout.strip().isdigit() else 0
    except Exception as e:
        logging.error("Pre-flight SSH command failed: %s", e)
        return 0

    if block_count == 0:
        logging.error("Pre-flight FAIL: Zero blocks in source subtitle %s", srt_path)
        return 0

    try:
        res_size = run_ssh(f"stat -c%s '{srt_path}'", host, user)
        file_size = int(res_size.stdout.strip())
    except Exception:
        file_size = 0

    if file_size < 1024:
        logging.error("Pre-flight FAIL: File size too small (%d bytes)", file_size)
        return 0

    logging.info("Pre-flight SUCCESS: %d blocks, %d bytes", block_count, file_size)
    return block_count

# ── Translation Core ──────────────────────────────────────────────────────────

async def translate_range(
    all_blocks: List[Dict[str, str]],
    start: int,
    end: int,
    worker_cfg: LocalAgentConfig,
    usage: Dict[str, Any],
    episode_api_calls: List[int],
    attempt: int = 1,
    corrective: str = "",
) -> Optional[List[Dict[str, str]]]:
    """
    Core recursive Worker translate range logic.
    If the worker fails, mismatches counts, or gets rejected, we auto-split the range in half
    and translate recursively, guaranteeing structural exactness.
    """
    block_count = end - start
    logging.info(
        "Translating range %d to %d (size %d blocks), attempt %d/%d",
        start, end, block_count, attempt, MAX_RETRIES,
    )

    if episode_api_calls[0] >= API_CALL_BUDGET:
        logging.error(
            "API call budget (%d) exhausted for this episode. Aborting range %d-%d.",
            API_CALL_BUDGET, start, end,
        )
        return None

    chunk_blocks = all_blocks[start:end]
    source_srt   = render_blocks(chunk_blocks, renumber=True)

    prompt = ""
    if corrective:
        prompt += f"CORRECTIVE GUIDANCE FROM PREVIOUS ATTEMPT:\n{corrective}\n\nPlease address the above issues in your new translation.\n\n---\n\n"
    prompt += "Translate the following SRT chunk from German to English:\n\n" + source_srt

    translated_raw = ""
    api_error = False
    try:
        async with Agent(worker_cfg) as worker:
            episode_api_calls[0] += 1
            response = await asyncio.wait_for(worker.chat(prompt), timeout=WORKER_TIMEOUT)
            translated_raw = await response.text()

            # Clean output before validation to prevent false rejection spirals
            translated_raw = strip_fences(translated_raw)
            translated_raw = translated_raw.replace('**', '')

            # Record tokens and cost
            usage_meta = response.usage_metadata
            if usage_meta:
                p_tokens   = usage_meta.prompt_token_count or 0
                ca_tokens  = usage_meta.cached_content_token_count or 0
                can_tokens = usage_meta.candidates_token_count or 0
                th_tokens  = usage_meta.thoughts_token_count or 0
                t_tokens   = usage_meta.total_token_count or 0

                non_cached_prompt = max(0, p_tokens - ca_tokens)
                input_cost  = (non_cached_prompt / 1_000_000) * PRICE_PER_M_INPUT
                cached_cost = (ca_tokens / 1_000_000) * PRICE_PER_M_CACHED
                output_cost = ((can_tokens + th_tokens) / 1_000_000) * PRICE_PER_M_OUTPUT
                chunk_cost  = input_cost + cached_cost + output_cost

                usage['prompt_tokens']     += p_tokens
                usage['cached_tokens']     += ca_tokens
                usage['candidates_tokens'] += can_tokens
                usage['thoughts_tokens']   += th_tokens
                usage['total_tokens']      += t_tokens
                usage['cost']              += chunk_cost

                logging.info(
                    "Range %d-%d usage: %d prompt (%d cached), %d candidates, %d thoughts. Est cost: $%.6f",
                    start, end, p_tokens, ca_tokens, can_tokens, th_tokens, chunk_cost,
                )
    except asyncio.TimeoutError:
        logging.warning("Worker call timed out after %ds for range %d-%d.", WORKER_TIMEOUT, start, end)
        api_error = True
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("quota", "rate limit", "429", "resource exhausted", "too many requests")):
            logging.error("Rate-limit/quota error for range %d-%d: %s. Aborting episode.", start, end, e)
            return None
        logging.warning("Worker API call failed: %s", e)
        api_error = True

    # Fallback / Split on Worker Failure
    if not translated_raw:
        if api_error:
            # Transport failure: retry with backoff only — never split into the quota storm
            if attempt < MAX_RETRIES:
                backoff = 2 ** attempt
                logging.info(
                    "Transport error. Backing off %ds, retrying range %d-%d, attempt %d",
                    backoff, start, end, attempt + 1,
                )
                await asyncio.sleep(backoff)
                return await translate_range(all_blocks, start, end, worker_cfg, usage, episode_api_calls, attempt + 1, corrective)
            else:
                logging.error("Transport error persists after %d attempts for range %d-%d. Aborting.", MAX_RETRIES, start, end)
                return None
        else:
            # Content failure: retry then split
            if attempt < MAX_RETRIES:
                logging.info("Worker failed (no output). Retrying range %d-%d, attempt %d", start, end, attempt + 1)
                return await translate_range(all_blocks, start, end, worker_cfg, usage, episode_api_calls, attempt + 1, "Worker returned empty output.")
            elif block_count > MIN_CHUNK_SIZE:
                mid = start + block_count // 2
                logging.info("Worker failed completely. Splitting range %d-%d into %d-%d and %d-%d", start, end, start, mid, mid, end)
                left = await translate_range(all_blocks, start, mid, worker_cfg, usage, episode_api_calls, attempt=1, corrective=corrective)
                if left is None: return None
                right = await translate_range(all_blocks, mid, end, worker_cfg, usage, episode_api_calls, attempt=1, corrective=corrective)
                if right is None: return None
                return left + right
            else:
                return None

    # Stage 1: Programmatic structural checks
    is_valid, err_msg = validate_translation_structure(translated_raw, chunk_blocks)
    if not is_valid:
        logging.warning("Programmatic validation failed for range %d-%d: %s", start, end, err_msg)
        if attempt < MAX_RETRIES:
            logging.info("Retrying range %d-%d, attempt %d", start, end, attempt + 1)
            return await translate_range(all_blocks, start, end, worker_cfg, usage, episode_api_calls, attempt + 1, err_msg)
        elif block_count > MIN_CHUNK_SIZE:
            mid = start + block_count // 2
            logging.info(
                "Splitting range %d-%d into %d-%d and %d-%d due to programmatic validation failure",
                start, end, start, mid, mid, end,
            )
            left = await translate_range(all_blocks, start, mid, worker_cfg, usage, episode_api_calls, attempt=1, corrective=err_msg)
            if left is None: return None
            right = await translate_range(all_blocks, mid, end, worker_cfg, usage, episode_api_calls, attempt=1, corrective=err_msg)
            if right is None: return None
            return left + right
        else:
            return None

    translated_blocks = parse_srt(translated_raw)

    # Map original timestamps to guarantee absolute correctness and wrap text
    for i, b in enumerate(translated_blocks):
        b['ts']   = chunk_blocks[i]['ts']
        b['text'] = wrap_subtitle_text(b['text'])
    return translated_blocks

# ── Process Episode ───────────────────────────────────────────────────────────

async def process_episode(
    episode_id: str,
    srt_path:   str,
    mkv_path:   str,
    cfg:        ShowConfig,
    dry_run:    bool,
    force:      bool,
    usage:      Dict[str, Any],
) -> None:
    """Full translation pipeline for one episode."""
    episode_api_calls = [0]
    logging.info("Starting processing for episode: %s", episode_id)

    sl = cfg.source_lang
    tl = cfg.target_lang

    # Determine output path alongside MKV (or derived from SRT name)
    if mkv_path:
        out_srt_path = os.path.join(
            os.path.dirname(mkv_path),
            os.path.basename(mkv_path).replace(".mkv", f".{tl}.srt"),
        )
    else:
        base = os.path.basename(srt_path)
        for ext in [f".{sl}.hi.synced.srt", f".{sl}.hi.srt", f".{sl}.srt", ".srt"]:
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        out_srt_path = os.path.join(os.path.dirname(srt_path), f"{base}.{tl}.srt")

    # Skip Check
    if not force:
        try:
            res = run_ssh(f"test -f '{out_srt_path}'", cfg.media_host, cfg.media_user, check=False)
            if res.returncode == 0:
                logging.info("SKIP: Output subtitle already exists at %s", out_srt_path)
                return
        except Exception:
            pass

    # Pre-flight
    source_blocks = preflight(srt_path, cfg.media_host, cfg.media_user)
    if source_blocks == 0:
        logging.error("Episode %s skipped because pre-flight failed", episode_id)
        return

    if dry_run:
        logging.info("DRY RUN: Pre-flight checks passed successfully for %s with %d blocks.", episode_id, source_blocks)
        return

    # Episodes small enough to fit in a single API call don't benefit from chunking
    chunk_size = cfg.chunk_size
    if source_blocks <= 60:
        chunk_size = source_blocks

    # Work Dir
    work_dir = tempfile.mkdtemp(prefix=f"subtitle-pipeline-{episode_id}-")
    logging.info("Created temporary working directory: %s", work_dir)

    # Resume or Init State
    state = load_state(episode_id, cfg.state_dir)
    if state:
        all_pending = all(c.get('status') == 'pending' for c in state.get('chunks', []))
        if all_pending and state.get('chunk_size') != chunk_size:
            logging.info(
                "Existing state for %s is completely pending but has chunk_size %s. Re-initializing with new chunk_size %d.",
                episode_id, state.get('chunk_size'), chunk_size,
            )
            state = init_state(episode_id, source_blocks, chunk_size, cfg.state_dir)
        else:
            logging.info("Resuming existing state.")
            state = resume_state(episode_id, cfg.state_dir)
    else:
        logging.info("Initializing fresh state.")
        state = init_state(episode_id, source_blocks, chunk_size, cfg.state_dir)

    # SCP Download Source SRT
    local_srt = os.path.join(work_dir, "source.srt")
    logging.info("Downloading source SRT to local path: %s", local_srt)
    download_file(srt_path, local_srt, cfg.media_host, cfg.media_user)

    # Parse blocks with Sig/BOM normalization
    with open(local_srt, "r", encoding="utf-8-sig") as f:
        srt_content = f.read()
    all_blocks   = parse_srt(srt_content)
    actual_blocks = len(all_blocks)
    logging.info("Parsed %d blocks locally.", actual_blocks)

    if actual_blocks != source_blocks:
        logging.warning(
            "Actual parsed blocks (%d) does not match preflight count (%d) — using parsed count.",
            actual_blocks, source_blocks,
        )
        source_blocks = actual_blocks
        if not state or state.get('source_blocks') != actual_blocks:
            logging.info("Re-initializing state for %s since block count changed or state is missing.", episode_id)
            state = init_state(episode_id, source_blocks, chunk_size, cfg.state_dir)
        else:
            logging.info("Preserving existing state since its source_blocks matches actual_blocks (%d).", actual_blocks)

    # Build Worker config once per episode from show's system prompt
    worker_cfg = make_worker_config(cfg.system_prompt)

    # Loop Chunks
    n_chunks = len(state['chunks'])
    failed   = False

    for i in range(n_chunks):
        chunk = state['chunks'][i]
        if chunk['status'] == 'done':
            logging.info("Chunk %d/%d: already completed (cached).", i, n_chunks - 1)
            continue

        blk_start = chunk['start']
        blk_end   = chunk['end']
        logging.info("Chunk %d/%d: Blocks %d to %d (size %d)", i, n_chunks - 1, blk_start + 1, blk_end, blk_end - blk_start)

        translated_chunk_blocks = await translate_range(
            all_blocks, blk_start, blk_end, worker_cfg, usage, episode_api_calls
        )
        if translated_chunk_blocks:
            chunk_out = chunk['output']
            with open(chunk_out, "w", encoding="utf-8") as f:
                f.write(render_blocks(translated_chunk_blocks, renumber=True))

            chunk['status'] = 'done'
            save_state(episode_id, state, cfg.state_dir)
            logging.info("Chunk %d/%d: Completed and saved.", i, n_chunks - 1)
        else:
            logging.error("Chunk %d: Failed unrecoverably! Aborting episode.", i)
            failed = True
            break

    if failed:
        logging.error("Episode %s failed. State has been preserved.", episode_id)
        return

    # Reassemble final file
    logging.info("Reassembling final SRT file.")
    final_blocks = []
    for chunk in state['chunks']:
        with open(chunk['output'], "r", encoding="utf-8") as f:
            chunk_content = f.read()
        final_blocks.extend(parse_srt(chunk_content))

    final_srt_local = os.path.join(work_dir, f"final.{tl}.srt")
    with open(final_srt_local, "w", encoding="utf-8") as f:
        f.write(render_blocks(final_blocks, renumber=True))

    # Final Structural Validation
    final_count = len(final_blocks)
    if final_count != source_blocks:
        logging.error(
            "FINAL VALIDATION FAIL: Final count (%d) != Source count (%d). Output retained in %s",
            final_count, source_blocks, work_dir,
        )
        return

    logging.info("FINAL VALIDATION SUCCESS: %d/%d blocks match perfectly.", final_count, source_blocks)

    # Upload
    logging.info("Uploading completed subtitles to %s at %s", cfg.media_host, out_srt_path)
    upload_file(final_srt_local, out_srt_path, cfg.media_host, cfg.media_user)

    # Complete State
    state['status'] = 'complete'
    save_state(episode_id, state, cfg.state_dir)
    logging.info("EPISODE COMPLETE: %s successfully deployed.", episode_id)

    # Cleanup temp directory and persisted per-chunk outputs (final file already deployed)
    shutil.rmtree(work_dir)
    for chunk in state['chunks']:
        try:
            if os.path.exists(chunk['output']):
                os.remove(chunk['output'])
        except OSError as e:
            logging.warning("Could not remove chunk output %s: %s", chunk['output'], e)
    logging.info("Cleaned up temporary working directory and per-chunk outputs.")
