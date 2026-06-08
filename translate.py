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
import json
import logging
import asyncio
import argparse
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from core.config import load_show, ShowConfig
from core.transfer import download_file, run_ssh
from core.pipeline import process_episode, log_usage_summary, make_usage_tracker
from core.backends.base import TranslationBackend
from core.backends.gemini import GeminiTranslationBackend
from core.backends.openai import OpenAITranslationBackend
from core.backends.anthropic import AnthropicTranslationBackend
from core.srt import parse_srt
from verify import check_srt, check_translations

DEFAULT_BAKEOFF_CONCURRENCY = 2


def _candidate_recommendation_key(candidate: Dict[str, Any]) -> Tuple[int, int, int, float]:
    structural_success = 1 if candidate.get("structural_success") else 0
    return (
        -structural_success,
        candidate.get("verifier_issue_count", 0),
        candidate.get("usage", {}).get("retry_count", 0) + candidate.get("usage", {}).get("split_count", 0),
        candidate.get("usage", {}).get("cost", 0.0),
    )


def _mark_recommendations(summary: Dict[str, Any]) -> None:
    candidates = [c for c in summary["candidates"] if not c.get("skipped")]
    for candidate in candidates:
        candidate["recommended_for_primary"] = False
        candidate["recommended_for_escalation"] = False
        candidate.setdefault("rejected_reason", "")

    if not candidates:
        return

    primary_candidates = [c for c in candidates if c.get("structural_success")]
    if primary_candidates:
        primary_winner = min(primary_candidates, key=_candidate_recommendation_key)
        primary_winner["recommended_for_primary"] = True

    escalation_candidates = [c for c in candidates if c.get("structural_success")]
    if escalation_candidates:
        escalation_winner = min(
            escalation_candidates,
            key=lambda c: (
                c.get("usage", {}).get("retry_count", 0) + c.get("usage", {}).get("split_count", 0),
                c.get("verifier_issue_count", 0),
                c.get("usage", {}).get("cost", 0.0),
            ),
        )
        escalation_winner["recommended_for_escalation"] = True

    for candidate in candidates:
        if candidate.get("recommended_for_primary") or candidate.get("recommended_for_escalation"):
            continue
        if not candidate.get("structural_success"):
            candidate["rejected_reason"] = "structural_failure"
        elif candidate.get("verifier_issue_count", 0) > 0:
            candidate["rejected_reason"] = "higher_verifier_issue_count"
        else:
            candidate["rejected_reason"] = "higher_cost_or_retry_count"


# ── Backend Factory ───────────────────────────────────────────────────────────

def _make_translation_backend(
    provider: str,
    model: str,
    system_prompt: str,
    thinking_level: str = "",
) -> TranslationBackend:
    provider = provider.lower()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            logging.critical("GEMINI_API_KEY not set in environment or .env")
            sys.exit(1)
        return GeminiTranslationBackend(key, system_prompt, model, thinking_level)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logging.critical("OPENAI_API_KEY not set in environment or .env")
            sys.exit(1)
        return OpenAITranslationBackend(key, system_prompt, model)
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            logging.critical("ANTHROPIC_API_KEY not set in environment or .env")
            sys.exit(1)
        return AnthropicTranslationBackend(key, system_prompt, model)
    raise ValueError(f"Unknown translation_backend: {provider!r}")


def _make_backends(cfg: ShowConfig) -> Tuple[TranslationBackend, Optional[TranslationBackend]]:
    primary = _make_translation_backend(
        cfg.translation_backend,
        cfg.translation_model,
        cfg.system_prompt,
        cfg.gemini_thinking_level,
    )
    escalation = None
    if cfg.escalation_model:
        escalation = _make_translation_backend(
            cfg.escalation_backend or cfg.translation_backend,
            cfg.escalation_model,
            cfg.system_prompt,
            cfg.escalation_thinking_level,
        )
    return primary, escalation

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
    p.add_argument('--bakeoff',    action='store_true', help='Run local-only bakeoff without uploading subtitles')
    p.add_argument('--candidate',  action='append', default=[], help='Bakeoff candidate as backend:model (repeatable)')
    p.add_argument('--bakeoff-dir', default=None, help='Optional local directory for bakeoff outputs')
    p.add_argument('--gemini-thinking-level', choices=['minimal', 'low', 'medium', 'high'], default=None,
                   help='Override Gemini thinking level for the primary translation backend')
    p.add_argument('--escalation-thinking-level', choices=['minimal', 'low', 'medium', 'high'], default=None,
                   help='Override Gemini thinking level for the escalation backend')
    p.add_argument('--force',      action='store_true', help='Re-translate even if target SRT already exists')
    p.add_argument('--chunk-size', type=int, default=None, help='Override chunk_size from YAML')
    p.add_argument('target',       help='"all", "SxxExx" (e.g. S02E01), or "Sxx" (e.g. S02)')
    return p.parse_args()


def _candidate_specs(raw_candidates: List[str]) -> List[Tuple[str, str]]:
    if raw_candidates:
        parsed = []
        for raw in raw_candidates:
            if ":" not in raw:
                raise ValueError(f"Invalid candidate {raw!r}; expected backend:model")
            backend, model = raw.split(":", 1)
            parsed.append((backend.strip(), model.strip()))
        return parsed
    return [
        ("gemini", "gemini-3.5-flash"),
        ("gemini", "gemini-3.1-pro-preview"),
        ("openai", "gpt-5.4"),
        ("anthropic", "claude-sonnet-4-6"),
    ]


def _matches_target(ep_id: str, target: str) -> bool:
    if target == "all":
        return True
    if re.fullmatch(r'S\d{2}E\d{2}', target):
        return ep_id == target
    if re.fullmatch(r'S\d{2}', target):
        return ep_id.startswith(target)
    raise ValueError(f"Unrecognized target format: {target}")


def _validate_bakeoff_target(target: str) -> None:
    if target == "all":
        return
    if re.fullmatch(r'S\d{2}E\d{2}', target):
        return
    if re.fullmatch(r'S\d{2}', target):
        return
    raise ValueError(
        "Bakeoff target must be 'all', a season like S01, or a single episode like S01E12, "
        f"got {target!r}"
    )


def _find_mkv_path(cfg: ShowConfig, ep_id: str) -> str:
    try:
        res_mkv = run_ssh(
            f"find '{cfg.media_dir}' -name '*{ep_id}*.mkv' 2>/dev/null | head -1",
            cfg.media_host, cfg.media_user, check=False,
        )
        return res_mkv.stdout.strip()
    except Exception:
        return ""


def _build_mkv_path_map(cfg: ShowConfig, episode_ids: List[str]) -> Dict[str, str]:
    target_ids = set(episode_ids)
    if not target_ids:
        return {}

    try:
        res = run_ssh(
            f"find '{cfg.media_dir}' -name '*.mkv' 2>/dev/null | sort",
            cfg.media_host,
            cfg.media_user,
            check=False,
        )
    except Exception:
        return {}

    mkv_map: Dict[str, str] = {}
    if res.returncode != 0:
        return mkv_map

    for path in (line.strip() for line in res.stdout.splitlines()):
        if not path:
            continue
        ep_match = re.search(r'S\d{2}E\d{2}', path)
        if not ep_match:
            continue
        ep_id = ep_match.group(0)
        if ep_id in target_ids and ep_id not in mkv_map:
            mkv_map[ep_id] = path
    return mkv_map


def _load_source_subtitle_text(cfg: ShowConfig, source_srt_path: str) -> str:
    fd, local_source = tempfile.mkstemp(prefix="subtitle-pipeline-source-", suffix=".srt")
    os.close(fd)
    try:
        download_file(source_srt_path, local_source, cfg.media_host, cfg.media_user)
        with open(local_source, "r", encoding="utf-8-sig") as f:
            return f.read()
    finally:
        if os.path.exists(local_source):
            os.remove(local_source)


def _verify_local_translation(
    cfg: ShowConfig,
    source_srt_path: str,
    en_srt_path: str,
    source_content: Optional[str] = None,
) -> List[str]:
    if source_content is None:
        source_content = _load_source_subtitle_text(cfg, source_srt_path)
    de_content = source_content
    with open(en_srt_path, "r", encoding="utf-8") as f:
        en_content = f.read()

    issues = []
    de_blocks = parse_srt(de_content)
    en_blocks = parse_srt(en_content)
    de_issues = check_srt(de_content, cfg.source_lang, cfg)
    en_issues = check_srt(en_content, cfg.target_lang, cfg)
    issues.extend(f"[de] {issue}" for issue in de_issues)
    issues.extend(f"[en] {issue}" for issue in en_issues)
    if de_blocks and en_blocks:
        issues.extend(f"[xlate] {issue}" for issue in check_translations(de_blocks, en_blocks, cfg))
    return issues


def _select_bakeoff_episodes(
    srt_paths: List[str],
    target: str,
    max_episodes: Optional[int] = None,
) -> List[Tuple[str, str]]:
    episodes: List[Tuple[str, str]] = []
    for srt_path in srt_paths:
        ep_match = re.search(r'S\d{2}E\d{2}', srt_path)
        if not ep_match:
            continue
        ep_id = ep_match.group(0)
        if _matches_target(ep_id, target):
            episodes.append((ep_id, srt_path))

    if max_episodes is not None:
        return episodes[:max(0, max_episodes)]
    return episodes


def _should_stop_bakeoff_candidate(
    candidate_rows: List[Dict[str, Any]],
    max_verifier_issues: Optional[int] = None,
) -> Tuple[bool, str]:
    if any(not row.get("success", False) for row in candidate_rows):
        return True, "structural_failure"

    if max_verifier_issues is not None:
        total_issues = sum(int(row.get("issue_count", 0)) for row in candidate_rows)
        if total_issues > max_verifier_issues:
            return True, "verifier_issue_budget_exceeded"

    return False, ""


def _candidate_efficiency_metrics(
    candidate_usage: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    total_episodes = len(candidate_rows)
    clean_episodes = sum(1 for row in candidate_rows if row.get("success") and not row.get("issues"))
    verifier_issues = sum(int(row.get("issue_count", 0)) for row in candidate_rows)
    api_calls = int(candidate_usage.get("api_calls", 0))
    cost = float(candidate_usage.get("cost", 0.0))

    return {
        "api_calls_per_episode": (api_calls / total_episodes) if total_episodes else None,
        "cost_per_clean_episode": (cost / clean_episodes) if clean_episodes else None,
        "verifier_issues_per_episode": (verifier_issues / total_episodes) if total_episodes else None,
        "clean_episode_rate": (clean_episodes / total_episodes) if total_episodes else None,
    }


async def _run_with_concurrency_limit(limit: int, coroutines: List[asyncio.Future]) -> List[Any]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _guard(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[_guard(coro) for coro in coroutines])


async def _run_bakeoff_candidate(
    cfg: ShowConfig,
    episodes: List[Tuple[str, str, str]],
    source_cache: Dict[str, str],
    bakeoff_root: Path,
    backend_name: str,
    model_name: str,
    force: bool,
    max_verifier_issues: Optional[int] = None,
) -> Dict[str, Any]:
    label = f"{backend_name}:{model_name}"
    if backend_name.lower() == "gemini" and cfg.gemini_thinking_level:
        label = f"{label}@{cfg.gemini_thinking_level}"
    candidate_dir = bakeoff_root / label.replace(":", "__")
    outputs_dir = candidate_dir / "outputs"
    state_dir = candidate_dir / "state"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    keep_escalation = (
        backend_name == cfg.translation_backend
        and model_name == cfg.translation_model
    )
    candidate_cfg = replace(
        cfg,
        translation_backend=backend_name,
        translation_model=model_name,
        escalation_backend=cfg.escalation_backend if keep_escalation else "",
        escalation_model=cfg.escalation_model if keep_escalation else "",
        escalation_thinking_level=cfg.escalation_thinking_level if keep_escalation else "",
        state_dir=str(state_dir),
    )

    try:
        primary_backend, escalation_backend = _make_backends(candidate_cfg)
    except SystemExit:
        logging.warning("Skipping bakeoff candidate %s due to missing credentials.", label)
        return {
            "candidate": label,
            "skipped": True,
            "reason": "missing_credentials",
        }

    candidate_usage = make_usage_tracker()
    candidate_rows = []
    stop_reason = ""

    for ep_id, srt_path, mkv_path in episodes:
        try:
            result = await process_episode(
                ep_id,
                srt_path,
                mkv_path,
                candidate_cfg,
                primary_backend,
                escalation_backend,
                dry_run=False,
                force=force,
                usage=candidate_usage,
                deploy=False,
                output_dir=str(outputs_dir),
                state_dir=str(state_dir),
                source_content=source_cache.get(srt_path),
            )
        except Exception as exc:
            logging.exception("Bakeoff candidate %s failed on %s: %s", label, ep_id, exc)
            candidate_rows.append({
                "episode": ep_id,
                "success": False,
                "output_path": None,
                "error": f"candidate_exception: {exc}",
                "issues": [],
                "issue_count": 0,
            })
            stop_reason = "candidate_exception"
            break
        issues = []
        if result.get("success") and result.get("output_path"):
            issues = _verify_local_translation(
                candidate_cfg,
                srt_path,
                result["output_path"],
                source_content=source_cache.get(srt_path),
            )
        candidate_rows.append({
            "episode": ep_id,
            "success": bool(result.get("success")),
            "output_path": result.get("output_path"),
            "error": result.get("error"),
            "issues": issues,
            "issue_count": len(issues),
        })
        should_stop, stop_reason = _should_stop_bakeoff_candidate(
            candidate_rows,
            max_verifier_issues=max_verifier_issues,
        )
        if should_stop:
            logging.info("Stopping bakeoff candidate %s early after %s.", label, stop_reason)
            break

    structural_success = all(row["success"] for row in candidate_rows)
    verifier_issue_count = sum(row["issue_count"] for row in candidate_rows)
    clean = sum(1 for row in candidate_rows if row["success"] and not row["issues"])
    total = len(candidate_rows)
    failure_class_counts = dict(candidate_usage.get("failure_classes", {}))
    used_escalation = bool(candidate_usage.get("escalation_count", 0) or candidate_usage.get("direct_to_escalation_chunks", 0))
    pricing_basis = getattr(primary_backend, "pricing_basis", None)
    efficiency_metrics = _candidate_efficiency_metrics(candidate_usage, candidate_rows)

    return {
        "candidate": label,
        "provider": backend_name,
        "model": model_name,
        "thinking_level": candidate_cfg.gemini_thinking_level if backend_name.lower() == "gemini" else "",
        "pricing_basis": pricing_basis,
        "clean": clean,
        "total_episodes": total,
        "structural_success": structural_success,
        "verifier_issue_count": verifier_issue_count,
        "failure_class_counts": failure_class_counts,
        "used_escalation": used_escalation,
        "escalation_successes": candidate_usage.get("escalated_chunk_successes", 0),
        "stopped_early": bool(stop_reason),
        "stop_reason": stop_reason,
        **efficiency_metrics,
        "usage": candidate_usage,
        "episodes": candidate_rows,
    }


async def _run_bakeoff(
    cfg: ShowConfig,
    target: str,
    force: bool,
    srt_paths: List[str],
    candidate_specs: List[Tuple[str, str]],
    bakeoff_dir: Optional[str],
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    bakeoff_root = Path(
        bakeoff_dir or f"{cfg.state_dir}/bakeoff/{timestamp}"
    )
    bakeoff_root.mkdir(parents=True, exist_ok=True)

    bakeoff_max_episodes_raw = os.environ.get("BAKEOFF_MAX_EPISODES", "").strip()
    bakeoff_max_episodes = (
        int(bakeoff_max_episodes_raw)
        if bakeoff_max_episodes_raw
        else None
    )

    selected_episodes = _select_bakeoff_episodes(
        srt_paths,
        target,
        max_episodes=bakeoff_max_episodes,
    )

    if not selected_episodes:
        logging.warning("No matching episodes found for bakeoff target=%s", target)
        return

    mkv_map = _build_mkv_path_map(cfg, [ep_id for ep_id, _ in selected_episodes])
    episodes: List[Tuple[str, str, str]] = []
    for ep_id, srt_path in selected_episodes:
        mkv_path = mkv_map.get(ep_id) or _find_mkv_path(cfg, ep_id)
        episodes.append((ep_id, srt_path, mkv_path))

    source_cache: Dict[str, str] = {}
    for ep_id, srt_path, _ in episodes:
        try:
            source_cache[srt_path] = _load_source_subtitle_text(cfg, srt_path)
        except Exception as exc:
            logging.warning("Could not prefetch source subtitle for %s (%s): %s", ep_id, srt_path, exc)

    summary: Dict[str, Any] = {
        "show": cfg.name,
        "target": target,
        "generated_at_utc": timestamp,
        "candidates": [],
    }

    bakeoff_concurrency = int(os.environ.get("BAKEOFF_CONCURRENCY", str(DEFAULT_BAKEOFF_CONCURRENCY)))
    bakeoff_max_verifier_issues_raw = os.environ.get("BAKEOFF_MAX_VERIFIER_ISSUES", "").strip()
    bakeoff_max_verifier_issues = (
        int(bakeoff_max_verifier_issues_raw)
        if bakeoff_max_verifier_issues_raw
        else None
    )
    summary["bakeoff_concurrency"] = max(1, bakeoff_concurrency)
    summary["bakeoff_max_episodes"] = bakeoff_max_episodes
    summary["bakeoff_max_verifier_issues"] = bakeoff_max_verifier_issues
    summary["candidates"] = await _run_with_concurrency_limit(
        summary["bakeoff_concurrency"],
        [
            _run_bakeoff_candidate(
                cfg,
                episodes,
                source_cache,
                bakeoff_root,
                backend_name,
                model_name,
                force,
                bakeoff_max_verifier_issues,
            )
            for backend_name, model_name in candidate_specs
        ],
    )

    _mark_recommendations(summary)
    summary_path = bakeoff_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Bakeoff summary written to {summary_path}")
    for candidate in summary["candidates"]:
        if candidate.get("skipped"):
            print(f"{candidate['candidate']}: skipped ({candidate['reason']})")
            continue
        print(
            f"{candidate['candidate']}: clean {candidate['clean']}/{candidate['total_episodes']}, "
            f"cost ${candidate['usage']['cost']:.6f}, "
            f"retries {candidate['usage']['retry_count']}, "
            f"escalations {candidate['usage']['escalation_count']}, "
            f"direct {candidate['usage'].get('direct_to_escalation_chunks', 0)}, "
            f"cost/clean {candidate.get('cost_per_clean_episode')}, "
            f"api/ep {candidate.get('api_calls_per_episode')}"
        )

# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    # Load config before logging is set up (config derives the log path)
    cfg = load_show(args.show)
    setup_logging(cfg.translate_log)

    # CLI flags take precedence; env vars as fallback
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.gemini_thinking_level is not None:
        cfg.gemini_thinking_level = args.gemini_thinking_level
    if args.escalation_thinking_level is not None:
        cfg.escalation_thinking_level = args.escalation_thinking_level
    dry_run = args.dry_run or os.environ.get('DRY_RUN', '0') == '1'
    force   = args.force   or os.environ.get('FORCE',   '0') == '1'
    target  = args.target

    logging.info(
        "=== Translation Run: show=%s, target=%s, DRY_RUN=%s, BAKEOFF=%s, FORCE=%s, CHUNK_SIZE=%d ===",
        cfg.name, target, dry_run, args.bakeoff, force, cfg.chunk_size,
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
        res = run_ssh(f"find '{cfg.media_dir}' -name '*.{cfg.source_lang}.srt' | sort",
                      cfg.media_host, cfg.media_user)
        srt_paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
    except Exception as e:
        logging.critical("Failed to search files on %s: %s", cfg.media_host, e)
        sys.exit(1)

    matching_episode_ids = []
    for srt_path in srt_paths:
        ep_match = re.search(r'S\d{2}E\d{2}', srt_path)
        if not ep_match:
            continue
        ep_id = ep_match.group(0)
        if _matches_target(ep_id, target):
            matching_episode_ids.append(ep_id)
            if dry_run:
                logging.info("DRY RUN match: %s (%s)", srt_path, ep_id)

    if args.bakeoff:
        _validate_bakeoff_target(target)
        await _run_bakeoff(cfg, target, force, srt_paths, _candidate_specs(args.candidate), args.bakeoff_dir)
        logging.info("=== Run complete ===")
        return

    # Build translation backend once per run
    backend, escalation_backend = _make_backends(cfg)
    mkv_map = _build_mkv_path_map(cfg, matching_episode_ids) if not dry_run else {}

    usage = make_usage_tracker()
    found_episodes = 0
    for srt_path in srt_paths:
        ep_match = re.search(r'S\d{2}E\d{2}', srt_path)
        if not ep_match:
            continue
        ep_id = ep_match.group(0)
        if not _matches_target(ep_id, target):
            continue
        found_episodes += 1
        if dry_run:
            continue

        mkv_path = mkv_map.get(ep_id) or _find_mkv_path(cfg, ep_id)

        try:
            await process_episode(
                ep_id, srt_path, mkv_path, cfg, backend, escalation_backend,
                dry_run=False, force=force, usage=usage,
            )
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
