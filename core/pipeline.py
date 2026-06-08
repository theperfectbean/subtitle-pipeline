#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/pipeline.py
Backend-agnostic pipeline — chunking, validation, retry/split-fallback, state
engine, and per-episode orchestration for both translation and transcription.
"""

import os
import json
import logging
import asyncio
import math
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .config import ShowConfig
from .srt import (
    parse_srt,
    render_blocks,
    sort_and_renumber,
    wrap_subtitle_text,
    strip_fences,
    validate_translation_structure,
    normalize_subtitle_text,
)
from .transfer import run_ssh, download_file, upload_file, with_retry
from .backends.base import (
    TranslationBackend, TranscriptionBackend,
    RateLimitError, ContextLengthError, TransientAPIError, TranscriptionError,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CHUNK_SIZE  = 1
MAX_RETRIES     = 3
API_CALL_BUDGET = 200   # hard ceiling on API calls per episode before aborting
WORKER_TIMEOUT  = 120   # seconds before a single model call is considered hung

# ── Usage Tracker ─────────────────────────────────────────────────────────────

def _format_terminology_guidance(cfg: ShowConfig) -> str:
    """Render a compact per-show glossary for chunk translation prompts."""
    if not cfg.terminology:
        return ""

    lines = [
        "Required terminology for this show:",
        "When a source term appears, you must use the listed target term exactly.",
    ]
    for source_term, target_term in cfg.terminology.items():
        if isinstance(target_term, list):
            rendered_target = " / ".join(str(item) for item in target_term)
        else:
            rendered_target = str(target_term)
        lines.append(f"- {source_term} -> {rendered_target}")
    return "\n".join(lines)

def make_usage_tracker() -> Dict[str, Any]:
    """Return a fresh usage accumulator dict."""
    return {
        'prompt_tokens':     0,
        'cached_tokens':     0,
        'candidates_tokens': 0,
        'thoughts_tokens':   0,
        'total_tokens':      0,
        'cost':              0.0,
        'retry_count':       0,
        'split_count':       0,
        'escalation_count':  0,
        'api_calls':         0,
        'structural_failures': 0,
        'failure_classes':   {},
        'primary_structural_failures': 0,
        'escalated_chunk_successes': 0,
        'direct_to_escalation_chunks': 0,
    }


def _classify_structural_failure(err_msg: str) -> str:
    lower_msg = err_msg.lower()
    if "leaked reasoning or metadata text detected" in lower_msg:
        return "reasoning_leak"
    if "preamble detected" in lower_msg:
        return "preamble"
    if "block count mismatch" in lower_msg:
        return "block_count"
    if "timestamp count mismatch" in lower_msg:
        return "timestamp_count"
    if "html or subtitle markup detected" in lower_msg:
        return "markup"
    if "markdown code fences" in lower_msg:
        return "markdown"
    if "output is completely empty" in lower_msg or "backend returned empty output" in lower_msg:
        return "empty_output"
    return "other_structural"


def _record_failure_class(usage: Dict[str, Any], failure_class: str) -> None:
    counts = usage.setdefault('failure_classes', {})
    counts[failure_class] = counts.get(failure_class, 0) + 1


def _should_immediately_escalate_structural_failure(err_msg: str) -> bool:
    return _classify_structural_failure(err_msg) in {"reasoning_leak", "preamble"}


def _should_retry_then_escalate_structural_failure(err_msg: str) -> bool:
    return _classify_structural_failure(err_msg) in {
        "block_count",
        "timestamp_count",
        "markup",
        "markdown",
        "empty_output",
    }


def _should_remember_chunk_as_known_bad(failure_class: str) -> bool:
    return failure_class in {
        "reasoning_leak",
        "preamble",
        "block_count",
        "timestamp_count",
        "markup",
        "markdown",
        "empty_output",
    }


def _should_start_chunk_on_escalation(
    chunk_state: Optional[Dict[str, Any]],
    escalation_backend: Optional[TranslationBackend],
) -> bool:
    if not escalation_backend or not chunk_state:
        return False
    failure_class = chunk_state.get('last_failure_class', "")
    return (
        chunk_state.get('preferred_backend') == 'escalation'
        and chunk_state.get('primary_failed_structurally', False)
        and chunk_state.get('escalation_succeeded', False)
        and _should_remember_chunk_as_known_bad(failure_class)
    )


def _max_primary_attempts_for_structural_failure(err_msg: str) -> int:
    if _should_immediately_escalate_structural_failure(err_msg):
        return 1
    if _should_retry_then_escalate_structural_failure(err_msg):
        return 2
    return MAX_RETRIES

def log_usage_summary(usage: Dict[str, Any]) -> None:
    """Log the cumulative token and cost summary for a translation run."""
    logging.info("=== Usage & Cost Summary ===")
    logging.info("Total Prompt Tokens:     %d", usage['prompt_tokens'])
    logging.info("Total Cached Tokens:     %d", usage['cached_tokens'])
    logging.info("Total Output Tokens:     %d", usage['candidates_tokens'])
    logging.info("Total Thoughts Tokens:   %d", usage['thoughts_tokens'])
    logging.info("Total Cumulative Tokens: %d", usage['total_tokens'])
    logging.info("Estimated Total Cost:    $%.6f", usage['cost'])
    logging.info("Total Retries:           %d", usage['retry_count'])
    logging.info("Total Splits:            %d", usage['split_count'])
    logging.info("Total Escalations:       %d", usage['escalation_count'])
    logging.info("Total API Calls:         %d", usage['api_calls'])
    logging.info("Primary Structural Failures: %d", usage['primary_structural_failures'])
    logging.info("Escalated Chunk Successes:   %d", usage['escalated_chunk_successes'])
    logging.info("Direct-to-Escalation Chunks: %d", usage['direct_to_escalation_chunks'])
    if usage.get('failure_classes'):
        logging.info("Failure Classes:         %s", json.dumps(usage['failure_classes'], sort_keys=True))
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
            'last_failure_class': "",
            'primary_failed_structurally': False,
            'escalation_succeeded': False,
            'preferred_backend': "",
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
        chunk.setdefault('last_failure_class', "")
        chunk.setdefault('primary_failed_structurally', False)
        chunk.setdefault('escalation_succeeded', False)
        chunk.setdefault('preferred_backend', "")
        if chunk['status'] == 'done' and not os.path.exists(chunk['output']):
            logging.warning("Cached chunk output file %s not found. Resetting to pending.", chunk['output'])
            chunk['status'] = 'pending'

    save_state(episode, state, state_dir)
    return state

# ── Pre-flight Checks ─────────────────────────────────────────────────────────

def preflight_local_srt(srt_path: str, srt_content: str, file_size: int) -> int:
    """Validate a downloaded source subtitle and return parsed block count."""
    if file_size < 1024:
        logging.error("Pre-flight FAIL: File size too small (%d bytes) for %s", file_size, srt_path)
        return 0

    blocks = parse_srt(srt_content)
    block_count = len(blocks)
    if block_count == 0:
        logging.error("Pre-flight FAIL: Zero parsed blocks in source subtitle %s", srt_path)
        return 0

    logging.info("Pre-flight SUCCESS: %d blocks, %d bytes", block_count, file_size)
    return block_count

# ── Translation Core ──────────────────────────────────────────────────────────

async def translate_range(
    all_blocks: List[Dict[str, str]],
    start: int,
    end: int,
    cfg: ShowConfig,
    primary_backend: TranslationBackend,
    escalation_backend: Optional[TranslationBackend],
    usage: Dict[str, Any],
    episode_api_calls: List[int],
    chunk_state: Optional[Dict[str, Any]] = None,
    attempt: int = 1,
    corrective: str = "",
    use_escalation: bool = False,
) -> Optional[List[Dict[str, str]]]:
    """
    Core recursive translate-range logic.
    If the backend fails, mismatches counts, or gets rejected, we auto-split the
    range in half and translate recursively, guaranteeing structural exactness.
    """
    block_count = end - start
    logging.info(
        "Translating range %d to %d (size %d blocks), attempt %d/%d via %s:%s",
        start, end, block_count, attempt, MAX_RETRIES,
        (escalation_backend if use_escalation and escalation_backend else primary_backend).provider,
        (escalation_backend if use_escalation and escalation_backend else primary_backend).model,
    )

    backend = escalation_backend if use_escalation and escalation_backend else primary_backend

    if episode_api_calls[0] >= API_CALL_BUDGET:
        logging.error(
            "API call budget (%d) exhausted for this episode. Aborting range %d-%d.",
            API_CALL_BUDGET, start, end,
        )
        return None

    chunk_blocks = all_blocks[start:end]
    source_srt   = render_blocks(chunk_blocks, renumber=True)
    terminology_guidance = _format_terminology_guidance(cfg)

    prompt = ""
    if corrective:
        prompt += f"CORRECTIVE GUIDANCE FROM PREVIOUS ATTEMPT:\n{corrective}\n\nPlease address the above issues in your new translation.\n\n---\n\n"
    if terminology_guidance:
        prompt += terminology_guidance + "\n\n"
    prompt += "Translate the following SRT chunk from German to English:\n\n" + source_srt

    translated_raw = ""
    api_error      = False

    try:
        episode_api_calls[0] += 1
        usage['api_calls'] += 1
        text, chunk_usage = await asyncio.wait_for(
            backend.translate(prompt), timeout=WORKER_TIMEOUT
        )

        # Clean output before validation to prevent false rejection spirals
        translated_raw = strip_fences(text)
        translated_raw = translated_raw.replace('**', '')

        # Accumulate usage
        usage['prompt_tokens']     += chunk_usage.prompt_tokens
        usage['cached_tokens']     += chunk_usage.cached_tokens
        usage['candidates_tokens'] += chunk_usage.output_tokens
        usage['thoughts_tokens']   += chunk_usage.thoughts_tokens
        usage['total_tokens']      += chunk_usage.total_tokens
        usage['cost']              += chunk_usage.cost

        logging.info(
            "Range %d-%d usage: %d prompt (%d cached), %d candidates, %d thoughts. Est cost: $%.6f",
            start, end,
            chunk_usage.prompt_tokens, chunk_usage.cached_tokens,
            chunk_usage.output_tokens, chunk_usage.thoughts_tokens,
            chunk_usage.cost,
        )

    except asyncio.TimeoutError:
        logging.warning("Backend call timed out after %ds for range %d-%d.", WORKER_TIMEOUT, start, end)
        api_error = True
    except RateLimitError as exc:
        logging.error("Rate-limit/quota error for range %d-%d: %s. Aborting episode.", start, end, exc)
        return None
    except ContextLengthError:
        # Context too long — split immediately, no point retrying
        if block_count > MIN_CHUNK_SIZE:
            mid = start + block_count // 2
            logging.info(
                "Context length error. Splitting range %d-%d into %d-%d and %d-%d",
                start, end, start, mid, mid, end,
            )
            usage['split_count'] += 1
            left = await translate_range(all_blocks, start, mid, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=corrective, use_escalation=use_escalation)
            if left is None: return None
            right = await translate_range(all_blocks, mid, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=corrective, use_escalation=use_escalation)
            if right is None: return None
            return left + right
        logging.error("Context length error on single-block range %d-%d. Aborting.", start, end)
        return None
    except (TransientAPIError, Exception) as exc:
        logging.warning("Backend API call failed: %s", exc)
        api_error = True

    # Fallback / Split on Backend Failure
    if not translated_raw:
        if api_error:
            # Transport failure: retry with backoff only — never split into the quota storm
            if attempt < MAX_RETRIES:
                backoff = 2 ** attempt
                usage['retry_count'] += 1
                logging.info(
                    "Transport error. Backing off %ds, retrying range %d-%d, attempt %d",
                    backoff, start, end, attempt + 1,
                )
                await asyncio.sleep(backoff)
                return await translate_range(all_blocks, start, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=attempt + 1, corrective=corrective, use_escalation=use_escalation)
            else:
                logging.error("Transport error persists after %d attempts for range %d-%d. Aborting.", MAX_RETRIES, start, end)
                return None
        else:
            # Content failure: retry then split
            if chunk_state is not None and not use_escalation:
                chunk_state['last_failure_class'] = "empty_output"
                chunk_state['primary_failed_structurally'] = True
            if attempt < MAX_RETRIES:
                _record_failure_class(usage, "empty_output")
                usage['retry_count'] += 1
                logging.info("Backend failed (no output). Retrying range %d-%d, attempt %d", start, end, attempt + 1)
                return await translate_range(all_blocks, start, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=attempt + 1, corrective="Backend returned empty output.", use_escalation=use_escalation)
            elif escalation_backend and not use_escalation:
                usage['escalation_count'] += 1
                logging.info("Escalating range %d-%d to %s:%s after empty output", start, end, escalation_backend.provider, escalation_backend.model)
                return await translate_range(all_blocks, start, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective="Backend returned empty output.", use_escalation=True)
            elif block_count > MIN_CHUNK_SIZE:
                mid = start + block_count // 2
                usage['split_count'] += 1
                logging.info("Backend failed completely. Splitting range %d-%d into %d-%d and %d-%d", start, end, start, mid, mid, end)
                left = await translate_range(all_blocks, start, mid, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=corrective, use_escalation=use_escalation)
                if left is None: return None
                right = await translate_range(all_blocks, mid, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=corrective, use_escalation=use_escalation)
                if right is None: return None
                return left + right
            else:
                return None

    # Stage 1: Programmatic structural checks
    is_valid, err_msg = validate_translation_structure(translated_raw, chunk_blocks)
    if not is_valid:
        usage['structural_failures'] += 1
        failure_class = _classify_structural_failure(err_msg)
        _record_failure_class(usage, failure_class)
        if not use_escalation:
            usage['primary_structural_failures'] += 1
        if chunk_state is not None and not use_escalation:
            chunk_state['last_failure_class'] = failure_class
            chunk_state['primary_failed_structurally'] = True
        logging.warning("Programmatic validation failed for range %d-%d: %s", start, end, err_msg)
        primary_attempt_limit = _max_primary_attempts_for_structural_failure(err_msg)
        if escalation_backend and not use_escalation and _should_immediately_escalate_structural_failure(err_msg):
            usage['escalation_count'] += 1
            logging.info(
                "Immediately escalating range %d-%d to %s:%s after reasoning/metadata leak",
                start, end, escalation_backend.provider, escalation_backend.model,
            )
            return await translate_range(
                all_blocks,
                start,
                end,
                cfg,
                primary_backend,
                escalation_backend,
                usage,
                episode_api_calls,
                chunk_state=chunk_state,
                attempt=1,
                corrective=err_msg,
                use_escalation=True,
            )
        elif not use_escalation and attempt < primary_attempt_limit:
            usage['retry_count'] += 1
            logging.info("Retrying range %d-%d on primary backend, attempt %d", start, end, attempt + 1)
            return await translate_range(all_blocks, start, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=attempt + 1, corrective=err_msg, use_escalation=use_escalation)
        elif use_escalation and attempt < 2:
            usage['retry_count'] += 1
            logging.info("Retrying range %d-%d on escalation backend, attempt %d", start, end, attempt + 1)
            return await translate_range(all_blocks, start, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=attempt + 1, corrective=err_msg, use_escalation=use_escalation)
        elif escalation_backend and not use_escalation:
            usage['escalation_count'] += 1
            logging.info(
                "Escalating range %d-%d to %s:%s after structural validation failure",
                start, end, escalation_backend.provider, escalation_backend.model,
            )
            return await translate_range(all_blocks, start, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=err_msg, use_escalation=True)
        elif block_count > MIN_CHUNK_SIZE:
            mid = start + block_count // 2
            usage['split_count'] += 1
            logging.info(
                "Splitting range %d-%d into %d-%d and %d-%d due to programmatic validation failure",
                start, end, start, mid, mid, end,
            )
            left = await translate_range(all_blocks, start, mid, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=err_msg, use_escalation=use_escalation)
            if left is None: return None
            right = await translate_range(all_blocks, mid, end, cfg, primary_backend, escalation_backend, usage, episode_api_calls, chunk_state=chunk_state, attempt=1, corrective=err_msg, use_escalation=use_escalation)
            if right is None: return None
            return left + right
        else:
            return None

    translated_blocks = parse_srt(translated_raw)

    # Map original timestamps to guarantee absolute correctness and wrap text
    for i, b in enumerate(translated_blocks):
        b['ts']   = chunk_blocks[i]['ts']
        b['text'] = wrap_subtitle_text(b['text'])
    if chunk_state is not None:
        if use_escalation:
            failure_class = chunk_state.get('last_failure_class', "")
            if _should_remember_chunk_as_known_bad(failure_class):
                chunk_state['escalation_succeeded'] = True
                chunk_state['preferred_backend'] = 'escalation'
            usage['escalated_chunk_successes'] += 1
        else:
            chunk_state['preferred_backend'] = 'primary'
            chunk_state['last_failure_class'] = ""
            chunk_state['primary_failed_structurally'] = False
            chunk_state['escalation_succeeded'] = False
    return translated_blocks

# ── Process Episode (Translation) ─────────────────────────────────────────────

async def process_episode(
    episode_id: str,
    srt_path:   str,
    mkv_path:   str,
    cfg:        ShowConfig,
    primary_backend: TranslationBackend,
    escalation_backend: Optional[TranslationBackend],
    dry_run:    bool,
    force:      bool,
    usage:      Dict[str, Any],
    deploy:     bool = True,
    output_dir: Optional[str] = None,
    state_dir:  Optional[str] = None,
) -> Dict[str, Any]:
    """Full translation pipeline for one episode."""
    episode_api_calls = [0]
    logging.info("Starting processing for episode: %s", episode_id)
    active_state_dir = state_dir or cfg.state_dir

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

    if not deploy:
        local_name = os.path.basename(out_srt_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_srt_path = os.path.join(output_dir, local_name)
        else:
            out_srt_path = os.path.join(os.getcwd(), local_name)

    # Skip Check
    if deploy and not force:
        try:
            res = run_ssh(f"test -f '{out_srt_path}'", cfg.media_host, cfg.media_user, check=False)
            if res.returncode == 0:
                logging.info("SKIP: Output subtitle already exists at %s", out_srt_path)
                return {"success": True, "skipped": True, "output_path": out_srt_path, "episode_id": episode_id}
        except Exception:
            pass
    elif not deploy and not force and os.path.exists(out_srt_path):
        logging.info("SKIP: Local bakeoff output already exists at %s", out_srt_path)
        return {"success": True, "skipped": True, "output_path": out_srt_path, "episode_id": episode_id}

    # Work Dir
    work_dir = tempfile.mkdtemp(prefix=f"subtitle-pipeline-{episode_id}-")
    logging.info("Created temporary working directory: %s", work_dir)

    # Download source once, then derive both preflight and parsing from the same local copy.
    local_srt = os.path.join(work_dir, "source.srt")
    logging.info("Downloading source SRT to local path: %s", local_srt)
    download_file(srt_path, local_srt, cfg.media_host, cfg.media_user)

    with open(local_srt, "r", encoding="utf-8-sig") as f:
        srt_content = f.read()
    file_size = os.path.getsize(local_srt)
    source_blocks = preflight_local_srt(srt_path, srt_content, file_size)
    if source_blocks == 0:
        logging.error("Episode %s skipped because pre-flight failed", episode_id)
        shutil.rmtree(work_dir)
        return {"success": False, "error": "preflight_failed", "episode_id": episode_id}

    if dry_run:
        logging.info("DRY RUN: Pre-flight checks passed successfully for %s with %d blocks.", episode_id, source_blocks)
        shutil.rmtree(work_dir)
        return {"success": True, "skipped": True, "output_path": None, "episode_id": episode_id}

    # Episodes small enough to fit in a single API call don't benefit from chunking
    chunk_size = cfg.chunk_size
    if source_blocks <= 60:
        chunk_size = source_blocks

    # Resume or Init State
    state = load_state(episode_id, active_state_dir)
    if state:
        all_pending = all(c.get('status') == 'pending' for c in state.get('chunks', []))
        if all_pending and state.get('chunk_size') != chunk_size:
            logging.info(
                "Existing state for %s is completely pending but has chunk_size %s. Re-initializing with new chunk_size %d.",
                episode_id, state.get('chunk_size'), chunk_size,
            )
            state = init_state(episode_id, source_blocks, chunk_size, active_state_dir)
        else:
            logging.info("Resuming existing state.")
            state = resume_state(episode_id, active_state_dir)
    else:
        logging.info("Initializing fresh state.")
        state = init_state(episode_id, source_blocks, chunk_size, active_state_dir)

    # Parse blocks with BOM normalization
    all_blocks    = parse_srt(srt_content)
    for block in all_blocks:
        block['text'] = normalize_subtitle_text(block['text'])
    actual_blocks = len(all_blocks)
    logging.info("Parsed %d blocks locally.", actual_blocks)

    if actual_blocks != source_blocks:
        logging.warning(
            "Actual parsed blocks (%d) does not match preflight count (%d) — using parsed count.",
            actual_blocks, source_blocks,
        )
        source_blocks = actual_blocks

    # Always validate state consistency against the blocks we actually parsed.
    # A stale state from a different SRT file (same episode ID, different block count)
    # must be re-initialized to prevent out-of-bounds chunk slices.
    if not state or state.get('source_blocks') != actual_blocks:
        logging.info(
            "State source_blocks (%s) doesn't match actual parsed blocks (%d). Re-initializing.",
            state.get('source_blocks') if state else 'none', actual_blocks,
        )
        state = init_state(episode_id, actual_blocks, chunk_size, active_state_dir)

    # Chunk Loop
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
        start_with_escalation = _should_start_chunk_on_escalation(chunk, escalation_backend)
        if start_with_escalation:
            usage['direct_to_escalation_chunks'] += 1
            logging.info(
                "Chunk %d/%d: routing directly to escalation backend due to prior structural failure memory.",
                i, n_chunks - 1,
            )

        translated_chunk_blocks = await translate_range(
            all_blocks,
            blk_start,
            blk_end,
            cfg,
            primary_backend,
            escalation_backend,
            usage,
            episode_api_calls,
            chunk_state=chunk,
            use_escalation=start_with_escalation,
        )
        if translated_chunk_blocks:
            chunk_out = chunk['output']
            with open(chunk_out, "w", encoding="utf-8") as f:
                f.write(render_blocks(translated_chunk_blocks, renumber=True))

            chunk['status'] = 'done'
            save_state(episode_id, state, active_state_dir)
            logging.info("Chunk %d/%d: Completed and saved.", i, n_chunks - 1)
        else:
            logging.error("Chunk %d: Failed unrecoverably! Aborting episode.", i)
            save_state(episode_id, state, active_state_dir)
            failed = True
            break

    if failed:
        logging.error("Episode %s failed. State has been preserved.", episode_id)
        return {"success": False, "error": "chunk_failed", "episode_id": episode_id, "output_path": None}

    # Reassemble final file
    logging.info("Reassembling final SRT file.")
    final_blocks = []
    for chunk in state['chunks']:
        with open(chunk['output'], "r", encoding="utf-8") as f:
            chunk_content = f.read()
        final_blocks.extend(parse_srt(chunk_content))

    # Global sort + renumber: corrects any ordering anomalies from split/retry
    final_blocks = sort_and_renumber(final_blocks)
    logging.info("Applied global sort and renumber pass (%d blocks).", len(final_blocks))

    final_srt_local = os.path.join(work_dir, f"final.{tl}.srt")
    with open(final_srt_local, "w", encoding="utf-8") as f:
        f.write(render_blocks(final_blocks, renumber=False))  # seq already set by sort_and_renumber

    # Final Structural Validation
    final_count = len(final_blocks)
    if final_count != source_blocks:
        logging.error(
            "FINAL VALIDATION FAIL: Final count (%d) != Source count (%d). Output retained in %s",
            final_count, source_blocks, work_dir,
        )
        return {"success": False, "error": "final_validation_failed", "episode_id": episode_id, "output_path": None}

    logging.info("FINAL VALIDATION SUCCESS: %d/%d blocks match perfectly.", final_count, source_blocks)

    # Upload
    if deploy:
        logging.info("Uploading completed subtitles to %s at %s", cfg.media_host, out_srt_path)
        upload_file(final_srt_local, out_srt_path, cfg.media_host, cfg.media_user)
    else:
        shutil.copy2(final_srt_local, out_srt_path)
        logging.info("Wrote local bakeoff subtitle to %s", out_srt_path)

    # Complete State
    state['status'] = 'complete'
    save_state(episode_id, state, active_state_dir)
    logging.info("EPISODE COMPLETE: %s successfully %s.", episode_id, "deployed" if deploy else "rendered locally")

    # Cleanup
    shutil.rmtree(work_dir)
    for chunk in state['chunks']:
        try:
            if os.path.exists(chunk['output']):
                os.remove(chunk['output'])
        except OSError as e:
            logging.warning("Could not remove chunk output %s: %s", chunk['output'], e)
    logging.info("Cleaned up temporary working directory and per-chunk outputs.")
    return {
        "success": True,
        "error": None,
        "episode_id": episode_id,
        "output_path": out_srt_path,
        "api_calls": episode_api_calls[0],
    }

# ── Audio Extraction ──────────────────────────────────────────────────────────

def extract_audio(mkv_path: str, wav_path: str) -> bool:
    """Extract first audio stream from MKV as lossless PCM WAV. Returns True on success."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", mkv_path,
         "-map", "0:a:0", "-acodec", "pcm_s16le", "-ar", "48000", "-f", "wav", wav_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logging.error("ffmpeg failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
        return False
    return True

# ── Process Episode (Transcription) ───────────────────────────────────────────

def transcribe_episode(
    mkv_basename: str,
    cfg:          ShowConfig,
    backend:      TranscriptionBackend,
) -> bool:
    """Full transcription pipeline for one episode. Returns True if source-lang SRT was produced."""
    sl         = cfg.source_lang
    mkv_remote = f"{cfg.media_dir}/{mkv_basename}.mkv"
    srt_remote = f"{cfg.media_dir}/{mkv_basename}.{sl}.srt"

    # Skip if already done
    check = run_ssh(f'test -f "{srt_remote}" && echo EXISTS || echo MISSING',
                    cfg.media_host, cfg.media_user, check=False)
    if check.returncode == 0 and "EXISTS" in check.stdout:
        logging.info("SKIP %s — .%s.srt already exists", mkv_basename, sl)
        return True

    logging.info("=== START: %s ===", mkv_basename)
    tmpdir    = tempfile.mkdtemp(prefix="subtitle-pipeline-aai-")
    _ep_name  = os.path.basename(mkv_basename)
    mkv_local = os.path.join(tmpdir, f"{_ep_name}.mkv")
    wav_local = os.path.join(tmpdir, f"{_ep_name}.wav")
    srt_local = os.path.join(tmpdir, f"{_ep_name}.{sl}.srt")

    try:
        # 1. Download MKV from media server
        logging.info("Downloading MKV from VM 113 …")
        download_file(mkv_remote, mkv_local, cfg.media_host, cfg.media_user)

        # 2. Extract audio
        logging.info("Extracting audio …")
        if not extract_audio(mkv_local, wav_local):
            logging.error("Audio extraction failed for %s — skipping", mkv_basename)
            return False

        # Free MKV disk space before transcription upload
        os.remove(mkv_local)

        # 3. Transcribe via backend
        logging.info("Submitting to AssemblyAI Universal-3 Pro …")
        prompt = cfg.assemblyai_prompt.strip() or None
        if prompt and prompt.startswith('#'):
            prompt = None

        try:
            srt_text = with_retry(
                "transcribe",
                lambda: backend.transcribe(wav_local, cfg.source_lang, prompt),
            )
        except TranscriptionError as exc:
            logging.error("Transcription returned error for %s: %s", mkv_basename, exc)
            return False
        except Exception as exc:
            logging.error("Transcription returned no output for %s: %s", mkv_basename, exc)
            return False

        if not srt_text:
            logging.error("Transcription returned no output for %s", mkv_basename)
            return False

        # 4. Write SRT locally, upload to media server
        with open(srt_local, "w", encoding="utf-8") as fh:
            fh.write(srt_text)

        logging.info("Uploading .%s.srt to VM 113 …", sl)
        upload_file(srt_local, srt_remote, cfg.media_host, cfg.media_user)

        # 5. Verify
        verify = run_ssh(f'ls -la "{srt_remote}"', cfg.media_host, cfg.media_user, check=False)
        if verify.returncode != 0:
            logging.error("Verification failed — .%s.srt not found on VM 113 after upload", sl)
            return False
        logging.info("Verified: %s", verify.stdout.strip())
        logging.info("=== DONE: %s ===", mkv_basename)
        return True

    except Exception as exc:
        logging.error("Unhandled error for %s: %s", mkv_basename, exc)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
