import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import core.pipeline as pipeline
from core.backends.base import TranslationBackend, TranslationUsage
from core.backends.gemini import _thinking_config_for_model
from core.backends.openai import OpenAITranslationBackend
from core.config import load_show
from core.pipeline import (
    _atomic_copy_local,
    _classify_structural_failure,
    _verify_deployed_target_srt,
    _max_primary_attempts_for_structural_failure,
    _should_immediately_escalate_structural_failure,
    _should_start_chunk_on_escalation,
    _should_retry_then_escalate_structural_failure,
    _upload_file_atomic,
    _verify_existing_target_srt,
    make_usage_tracker,
    process_episode,
    translate_range,
)
import translate
from translate import (
    _build_mkv_path_map,
    _candidate_efficiency_metrics,
    _mark_recommendations,
    _run_bakeoff_candidate,
    _select_bakeoff_episodes,
    _merge_usage_totals,
    _should_stop_bakeoff_candidate,
    _run_with_concurrency_limit,
    _resolve_concurrency,
    _validate_bakeoff_target,
    _verify_local_translation,
)


class _FakeBackend(TranslationBackend):
    def __init__(self, provider, model, outputs):
        self._provider = provider
        self._model = model
        self._outputs = list(outputs)
        self.calls = 0

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens_per_chunk(self) -> int:
        return 8192

    async def translate(self, prompt: str):
        self.calls += 1
        text = self._outputs.pop(0)
        return text, TranslationUsage()


def _srt_text(block_count, text_prefix="Translated line", long_text=False):
    blocks = []
    for i in range(1, block_count + 1):
        start = i
        end = i + 1
        text = f"{text_prefix} {i}."
        if long_text:
            text += " " + ("This sentence keeps the source subtitle comfortably above preflight size. " * 8)
        blocks.append(
            f"{i}\n"
            f"00:00:{start:02d},000 --> 00:00:{end:02d},000\n"
            f"{text}\n"
        )
    return "\n".join(blocks) + "\n"


class ModelPolicyTests(unittest.TestCase):
    def test_gemini_35_flash_uses_minimal_thinking_level(self):
        cfg = _thinking_config_for_model("gemini-3.5-flash")
        self.assertEqual("MINIMAL", str(cfg.thinking_level).split(".")[-1])
        self.assertIsNone(cfg.thinking_budget)

    def test_gemini_31_pro_uses_low_thinking_level(self):
        cfg = _thinking_config_for_model("gemini-3.1-pro-preview")
        self.assertEqual("LOW", str(cfg.thinking_level).split(".")[-1])
        self.assertIsNone(cfg.thinking_budget)

    def test_explicit_thinking_override_wins(self):
        cfg = _thinking_config_for_model("gemini-3.5-flash", "low")
        self.assertEqual("LOW", str(cfg.thinking_level).split(".")[-1])
        self.assertIsNone(cfg.thinking_budget)

    def test_gemini_25_flash_keeps_zero_thinking_budget(self):
        cfg = _thinking_config_for_model("gemini-2.5-flash")
        self.assertEqual(0, cfg.thinking_budget)

    def test_openai_mini_uses_mini_pricing(self):
        backend = OpenAITranslationBackend("test-key", "Translate.", "gpt-5.4-mini")
        usage = backend._extract_usage({
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "prompt_tokens_details": {"cached_tokens": 0},
        })
        self.assertAlmostEqual(5.25, usage.cost)

    def test_bakeoff_requires_single_episode_target(self):
        _validate_bakeoff_target("S01E12")
        _validate_bakeoff_target("S01")
        _validate_bakeoff_target("all")

        with self.assertRaises(ValueError):
            _validate_bakeoff_target("episode-1")

    def test_verify_local_translation_uses_cached_source_content(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        en_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hello there.\n\n"
        )
        de_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo da.\n\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            en_path = os.path.join(tmpdir, "sample.en.srt")
            with open(en_path, "w", encoding="utf-8") as fh:
                fh.write(en_content)

            original_download = translate.download_file

            def _should_not_run(*args, **kwargs):
                raise AssertionError("download_file should not be called when source_content is supplied")

            translate.download_file = _should_not_run
            try:
                issues = _verify_local_translation(
                    cfg,
                    "/remote/sample.de.srt",
                    en_path,
                    source_content=de_content,
                )
            finally:
                translate.download_file = original_download

        self.assertTrue(all("Block count too low" in issue for issue in issues))

    def test_verify_local_translation_still_runs_translation_checks_with_warnings(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        de_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo da.\n\n"
        )
        en_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hello there.\n\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            en_path = os.path.join(tmpdir, "sample.en.srt")
            with open(en_path, "w", encoding="utf-8") as fh:
                fh.write(en_content)

            original_check_translations = translate.check_translations
            calls = []

            def _fake_check_translations(*args, **kwargs):
                calls.append((args, kwargs))
                return ["translation-level issue"]

            translate.check_translations = _fake_check_translations
            try:
                issues = _verify_local_translation(
                    cfg,
                    "/remote/sample.de.srt",
                    en_path,
                    source_content=de_content,
                )
            finally:
                translate.check_translations = original_check_translations

        self.assertTrue(any("Block count too low" in issue for issue in issues))
        self.assertIn("[xlate] translation-level issue", issues)
        self.assertEqual(1, len(calls))

    def test_verify_existing_target_srt_checks_local_output(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = os.path.join(tmpdir, "valid.en.srt")
            invalid_path = os.path.join(tmpdir, "invalid.en.srt")
            with open(valid_path, "w", encoding="utf-8") as fh:
                fh.write(_srt_text(11, "Clean English sentence"))
            with open(invalid_path, "w", encoding="utf-8") as fh:
                fh.write(_srt_text(11, "Bad<br/>markup"))

            self.assertEqual([], _verify_existing_target_srt(valid_path, cfg, deploy=False))
            self.assertTrue(_verify_existing_target_srt(invalid_path, cfg, deploy=False))

    def test_process_episode_regenerates_invalid_existing_local_output(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        source_content = _srt_text(11, "Deutsche Quelle", long_text=True)
        translated_content = _srt_text(11, "Clean English sentence")
        backend = _FakeBackend("gemini", "gemini-test", [translated_content])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            state_dir = os.path.join(tmpdir, "state")
            os.makedirs(output_dir)
            os.makedirs(state_dir)
            output_path = os.path.join(output_dir, "Sample.S01E99.en.srt")
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(_srt_text(11, "Bad<br/>markup"))

            result = asyncio.run(
                process_episode(
                    "S01E99",
                    "/remote/Sample.S01E99.de.srt",
                    "/remote/Sample.S01E99.mkv",
                    cfg,
                    backend,
                    None,
                    dry_run=False,
                    force=False,
                    usage=make_usage_tracker(),
                    deploy=False,
                    output_dir=output_dir,
                    state_dir=state_dir,
                    source_content=source_content,
                )
            )

            self.assertTrue(result["success"])
            self.assertFalse(result.get("skipped", False))
            self.assertEqual(1, backend.calls)
            with open(output_path, "r", encoding="utf-8") as fh:
                self.assertNotIn("<br/>", fh.read())

    def test_process_episode_blocks_final_output_when_target_verifier_fails(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        source_content = _srt_text(11, "Deutsche Quelle", long_text=True)
        translated_content = _srt_text(11, "Clean English sentence")
        backend = _FakeBackend("gemini", "gemini-test", [translated_content])
        original_target_srt_issues = pipeline._target_srt_issues

        pipeline._target_srt_issues = lambda content, cfg: ["forced final verification issue"]
        work_dir = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = os.path.join(tmpdir, "out")
                state_dir = os.path.join(tmpdir, "state")
                os.makedirs(output_dir)
                os.makedirs(state_dir)

                result = asyncio.run(
                    process_episode(
                        "S01E98",
                        "/remote/Sample.S01E98.de.srt",
                        "/remote/Sample.S01E98.mkv",
                        cfg,
                        backend,
                        None,
                        dry_run=False,
                        force=True,
                        usage=make_usage_tracker(),
                        deploy=False,
                        output_dir=output_dir,
                        state_dir=state_dir,
                        source_content=source_content,
                    )
                )
                work_dir = result.get("work_dir")
                self.assertFalse(result["success"])
                self.assertEqual("final_target_verification_failed", result["error"])
                self.assertEqual(["forced final verification issue"], result["issues"])
                self.assertFalse(os.path.exists(os.path.join(output_dir, "Sample.S01E98.en.srt")))
                self.assertTrue(work_dir and os.path.isdir(work_dir))
        finally:
            pipeline._target_srt_issues = original_target_srt_issues
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    def test_process_episode_does_not_resume_metadata_less_stale_state(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        source_content = _srt_text(11, "Neue deutsche Quelle", long_text=True)
        translated_content = _srt_text(11, "Fresh English sentence")
        backend = _FakeBackend("gemini", "gemini-test", [translated_content])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "out")
            state_dir = os.path.join(tmpdir, "state")
            os.makedirs(output_dir)
            os.makedirs(state_dir)
            stale_chunk_path = os.path.join(state_dir, "S01E97_chunk_0_final.srt")
            with open(stale_chunk_path, "w", encoding="utf-8") as fh:
                fh.write(_srt_text(11, "Stale English sentence"))
            with open(os.path.join(state_dir, "S01E97.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "episode": "S01E97",
                        "source_blocks": 11,
                        "chunk_size": 11,
                        "chunks": [
                            {
                                "id": 0,
                                "start": 0,
                                "end": 11,
                                "status": "done",
                                "output": stale_chunk_path,
                            }
                        ],
                        "status": "in_progress",
                    },
                    fh,
                )

            result = asyncio.run(
                process_episode(
                    "S01E97",
                    "/remote/Sample.S01E97.de.srt",
                    "/remote/Sample.S01E97.mkv",
                    cfg,
                    backend,
                    None,
                    dry_run=False,
                    force=True,
                    usage=make_usage_tracker(),
                    deploy=False,
                    output_dir=output_dir,
                    state_dir=state_dir,
                    source_content=source_content,
                )
            )

            self.assertTrue(result["success"])
            self.assertEqual(1, backend.calls)
            with open(os.path.join(output_dir, "Sample.S01E97.en.srt"), "r", encoding="utf-8") as fh:
                rendered = fh.read()
            self.assertIn("Fresh English sentence", rendered)
            self.assertNotIn("Stale English sentence", rendered)

    def test_atomic_copy_local_replaces_destination_after_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "src.srt")
            dest_path = os.path.join(tmpdir, "dest.srt")
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write("new subtitles")
            with open(dest_path, "w", encoding="utf-8") as fh:
                fh.write("old subtitles")

            _atomic_copy_local(src_path, dest_path)

            with open(dest_path, "r", encoding="utf-8") as fh:
                self.assertEqual("new subtitles", fh.read())
            self.assertFalse(any(name.endswith(f".codex-upload-{os.getpid()}") for name in os.listdir(tmpdir)))

    def test_upload_file_atomic_promotes_uploaded_temp_file(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        original_upload_file = pipeline.upload_file
        original_run_ssh = pipeline.run_ssh
        calls = []

        def _fake_upload_file(local_path, remote_path, host, user):
            calls.append(("upload", local_path, remote_path, host, user))

        def _fake_run_ssh(cmd, host, user, check=True):
            calls.append(("ssh", cmd, host, user, check))

            class Result:
                returncode = 0
                stdout = ""

            return Result()

        pipeline.upload_file = _fake_upload_file
        pipeline.run_ssh = _fake_run_ssh
        try:
            _upload_file_atomic("/tmp/final.en.srt", "/remote/final.en.srt", cfg)
        finally:
            pipeline.upload_file = original_upload_file
            pipeline.run_ssh = original_run_ssh

        self.assertEqual("upload", calls[0][0])
        self.assertTrue(calls[0][2].startswith("/remote/final.en.srt.codex-upload-"))
        self.assertEqual(("ssh",), (calls[1][0],))
        self.assertIn("mv -f", calls[1][1])
        self.assertIn(calls[0][2], calls[1][1])
        self.assertIn("/remote/final.en.srt", calls[1][1])
        self.assertEqual(("ssh",), (calls[2][0],))
        self.assertIn("rm -f", calls[2][1])
        self.assertFalse(calls[2][4])

    def test_verify_deployed_target_srt_accepts_matching_remote_file(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        original_run_ssh = pipeline.run_ssh

        class _Result:
            def __init__(self, stdout, returncode=0, stderr=""):
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = stderr

        def _fake_run_ssh(cmd, host, user, check=False):
            if "stat -c %s" in cmd:
                return _Result("11\n")
            if "sha256sum" in cmd:
                return _Result("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9\n")
            raise AssertionError(f"Unexpected SSH command: {cmd}")

        pipeline.run_ssh = _fake_run_ssh
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "final.en.srt")
                with open(local_path, "w", encoding="utf-8") as fh:
                    fh.write("hello world")
                self.assertEqual([], _verify_deployed_target_srt(local_path, "/remote/final.en.srt", cfg))
        finally:
            pipeline.run_ssh = original_run_ssh

    def test_verify_deployed_target_srt_reports_hash_mismatch(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        original_run_ssh = pipeline.run_ssh

        class _Result:
            def __init__(self, stdout, returncode=0, stderr=""):
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = stderr

        def _fake_run_ssh(cmd, host, user, check=False):
            if "stat -c %s" in cmd:
                return _Result("11\n")
            if "sha256sum" in cmd:
                return _Result("different-hash\n")
            raise AssertionError(f"Unexpected SSH command: {cmd}")

        pipeline.run_ssh = _fake_run_ssh
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "final.en.srt")
                with open(local_path, "w", encoding="utf-8") as fh:
                    fh.write("hello world")
                issues = _verify_deployed_target_srt(local_path, "/remote/final.en.srt", cfg)
        finally:
            pipeline.run_ssh = original_run_ssh

        self.assertEqual(1, len(issues))
        self.assertIn("Deployed hash mismatch", issues[0])

    def test_resolve_concurrency_prefers_cli_and_clamps_to_one(self):
        original_value = os.environ.get("TRANSLATE_CONCURRENCY")
        os.environ["TRANSLATE_CONCURRENCY"] = "5"
        try:
            self.assertEqual(3, _resolve_concurrency(3))
            self.assertEqual(1, _resolve_concurrency(0))
        finally:
            if original_value is None:
                os.environ.pop("TRANSLATE_CONCURRENCY", None)
            else:
                os.environ["TRANSLATE_CONCURRENCY"] = original_value

    def test_resolve_concurrency_uses_environment_fallback(self):
        original_value = os.environ.get("TRANSLATE_CONCURRENCY")
        try:
            os.environ["TRANSLATE_CONCURRENCY"] = "4"
            self.assertEqual(4, _resolve_concurrency(None))
            os.environ["TRANSLATE_CONCURRENCY"] = "0"
            self.assertEqual(1, _resolve_concurrency(None))
            os.environ["TRANSLATE_CONCURRENCY"] = "bad"
            self.assertEqual(1, _resolve_concurrency(None))
            os.environ.pop("TRANSLATE_CONCURRENCY", None)
            self.assertEqual(1, _resolve_concurrency(None))
        finally:
            if original_value is None:
                os.environ.pop("TRANSLATE_CONCURRENCY", None)
            else:
                os.environ["TRANSLATE_CONCURRENCY"] = original_value

    def test_merge_usage_totals_accumulates_failure_classes(self):
        target = make_usage_tracker()
        source = make_usage_tracker()
        source["prompt_tokens"] = 10
        source["cached_tokens"] = 2
        source["cost"] = 1.25
        source["retry_count"] = 3
        source["failure_classes"] = {"markup": 2, "block_count": 1}

        _merge_usage_totals(target, source)

        self.assertEqual(10, target["prompt_tokens"])
        self.assertEqual(2, target["cached_tokens"])
        self.assertEqual(1.25, target["cost"])
        self.assertEqual(3, target["retry_count"])
        self.assertEqual({"markup": 2, "block_count": 1}, target["failure_classes"])

    def test_select_bakeoff_episodes_respects_target_and_limit(self):
        episodes = _select_bakeoff_episodes(
            [
                "/media/show/Show.S01E01.de.srt",
                "/media/show/Show.S01E02.de.srt",
                "/media/show/Show.S02E01.de.srt",
            ],
            "S01",
            max_episodes=1,
        )

        self.assertEqual([("S01E01", "/media/show/Show.S01E01.de.srt")], episodes)

    def test_select_bakeoff_episodes_supports_all_target(self):
        episodes = _select_bakeoff_episodes(
            [
                "/media/show/Show.S01E01.de.srt",
                "/media/show/Show.S01E02.de.srt",
            ],
            "all",
        )

        self.assertEqual(
            [
                ("S01E01", "/media/show/Show.S01E01.de.srt"),
                ("S01E02", "/media/show/Show.S01E02.de.srt"),
            ],
            episodes,
        )

    def test_build_mkv_path_map_uses_single_listing(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        calls = []
        original_run_ssh = translate.run_ssh

        class _Result:
            returncode = 0
            stdout = (
                "/media/Show/Show.S01E01.mkv\n"
                "/media/Show/Show.S01E02.mkv\n"
                "/media/Other/NoEpisodeHere.mkv\n"
                "/media/Show/Show.S01E01.alt.mkv\n"
            )

        def _fake_run_ssh(cmd, host, user, check=False):
            calls.append((cmd, host, user, check))
            return _Result()

        translate.run_ssh = _fake_run_ssh
        try:
            mkv_map = _build_mkv_path_map(cfg, ["S01E01", "S01E02", "S01E03"])
        finally:
            translate.run_ssh = original_run_ssh

        self.assertEqual(1, len(calls))
        self.assertIn("find", calls[0][0])
        self.assertEqual(
            {
                "S01E01": "/media/Show/Show.S01E01.mkv",
                "S01E02": "/media/Show/Show.S01E02.mkv",
            },
            mkv_map,
        )

    def test_build_mkv_path_map_returns_empty_on_failure(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        original_run_ssh = translate.run_ssh

        def _fake_run_ssh(cmd, host, user, check=False):
            raise RuntimeError("ssh failed")

        translate.run_ssh = _fake_run_ssh
        try:
            mkv_map = _build_mkv_path_map(cfg, ["S01E01"])
        finally:
            translate.run_ssh = original_run_ssh

        self.assertEqual({}, mkv_map)

    def test_run_with_concurrency_limit_bounds_parallelism(self):
        active = 0
        max_active = 0

        async def _job(value):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return value

        results = asyncio.run(
            _run_with_concurrency_limit(2, [_job(1), _job(2), _job(3), _job(4)])
        )

        self.assertEqual([1, 2, 3, 4], results)
        self.assertEqual(2, max_active)

    def test_should_stop_bakeoff_candidate_on_first_failed_episode(self):
        stop, reason = _should_stop_bakeoff_candidate(
            [
                {"episode": "S01E01", "success": True, "issue_count": 0},
                {"episode": "S01E02", "success": False, "issue_count": 0},
            ]
        )

        self.assertTrue(stop)
        self.assertEqual("structural_failure", reason)

    def test_should_stop_bakeoff_candidate_on_verifier_issue_budget(self):
        stop, reason = _should_stop_bakeoff_candidate(
            [
                {"episode": "S01E01", "success": True, "issue_count": 1},
                {"episode": "S01E02", "success": True, "issue_count": 2},
            ],
            max_verifier_issues=2,
        )

        self.assertTrue(stop)
        self.assertEqual("verifier_issue_budget_exceeded", reason)

    def test_should_not_stop_bakeoff_candidate_when_still_within_budget(self):
        stop, reason = _should_stop_bakeoff_candidate(
            [
                {"episode": "S01E01", "success": True, "issue_count": 1},
                {"episode": "S01E02", "success": True, "issue_count": 1},
            ],
            max_verifier_issues=2,
        )

        self.assertFalse(stop)
        self.assertEqual("", reason)

    def test_candidate_efficiency_metrics_computes_expected_ratios(self):
        metrics = _candidate_efficiency_metrics(
            {"api_calls": 9, "cost": 3.0},
            [
                {"success": True, "issues": [], "issue_count": 0},
                {"success": True, "issues": [], "issue_count": 1},
                {"success": False, "issues": ["x"], "issue_count": 2},
            ],
        )

        self.assertAlmostEqual(3.0, metrics["api_calls_per_episode"])
        self.assertAlmostEqual(1.5, metrics["cost_per_clean_episode"])
        self.assertAlmostEqual(1.0, metrics["verifier_issues_per_episode"])
        self.assertAlmostEqual(2 / 3, metrics["clean_episode_rate"])

    def test_run_bakeoff_candidate_passes_cached_source_content(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        original_make_backends = translate._make_backends
        original_process_episode = translate.process_episode
        original_verify = translate._verify_local_translation
        calls = []

        async def _fake_process_episode(*args, **kwargs):
            calls.append(kwargs.get("source_content"))
            return {
                "success": True,
                "output_path": "/tmp/out.en.srt",
                "error": None,
            }

        translate._make_backends = lambda candidate_cfg: (_FakeBackend("gemini", "gemini-test", []), None)
        translate.process_episode = _fake_process_episode
        translate._verify_local_translation = lambda *args, **kwargs: []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = asyncio.run(
                    _run_bakeoff_candidate(
                        cfg,
                        [("S01E01", "/remote/S01E01.de.srt", "/remote/S01E01.mkv")],
                        {"/remote/S01E01.de.srt": "cached source"},
                        Path(tmpdir),
                        "gemini",
                        "gemini-test",
                        force=True,
                    )
                )
        finally:
            translate._make_backends = original_make_backends
            translate.process_episode = original_process_episode
            translate._verify_local_translation = original_verify

        self.assertEqual(["cached source"], calls)
        self.assertTrue(result["structural_success"])

    def test_run_bakeoff_candidate_records_candidate_exceptions(self):
        cfg = load_show("/home/admin/subtitle-pipeline/shows/pumuckl-1982.yaml")
        original_make_backends = translate._make_backends
        original_process_episode = translate.process_episode

        async def _fake_process_episode(*args, **kwargs):
            raise RuntimeError("boom")

        translate._make_backends = lambda candidate_cfg: (_FakeBackend("gemini", "gemini-test", []), None)
        translate.process_episode = _fake_process_episode
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = asyncio.run(
                    _run_bakeoff_candidate(
                        cfg,
                        [("S01E01", "/remote/S01E01.de.srt", "/remote/S01E01.mkv")],
                        {},
                        Path(tmpdir),
                        "gemini",
                        "gemini-test",
                        force=True,
                    )
                )
        finally:
            translate._make_backends = original_make_backends
            translate.process_episode = original_process_episode

        self.assertFalse(result["structural_success"])
        self.assertTrue(result["stopped_early"])
        self.assertEqual("candidate_exception", result["stop_reason"])
        self.assertIn("candidate_exception", result["episodes"][0]["error"])

    def test_load_show_prefers_translation_model_and_defaults_judge_to_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "show.yaml")
            with open(config_path, "w", encoding="utf-8") as fh:
                fh.write(
                    'name: "Test Show"\n'
                    'media_dir: "/tmp/media"\n'
                    'source_lang: "de"\n'
                    'target_lang: "en"\n'
                    'system_prompt: "Translate."\n'
                    'translation_backend: "gemini"\n'
                    'gemini_model: "legacy-model"\n'
                    'translation_model: "preferred-model"\n'
                )

            cfg = load_show(config_path)

            self.assertEqual("preferred-model", cfg.translation_model)
            self.assertEqual("legacy-model", cfg.gemini_model)
            self.assertEqual("preferred-model", cfg.llm_judge_model)

    def test_translate_range_escalates_after_structural_failure(self):
        blocks = [
            {
                "seq": "1",
                "ts": "00:00:01,000 --> 00:00:02,000",
                "text": "Hallo",
            }
        ]
        primary = _FakeBackend("gemini", "gemini-3.5-flash", ["Hello<br/>there", "Hello<br/>there"])
        escalation = _FakeBackend(
            "gemini",
            "gemini-3.1-pro-preview",
            [
                "1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n",
            ],
        )
        usage = make_usage_tracker()

        class Cfg:
            terminology = {}

        result = asyncio.run(
            translate_range(
                blocks,
                0,
                1,
                Cfg(),
                primary,
                escalation,
                usage,
                [0],
            )
        )

        self.assertEqual(2, primary.calls)
        self.assertEqual(1, escalation.calls)
        self.assertEqual(1, usage["escalation_count"])
        self.assertEqual("Hello there.", result[0]["text"])

    def test_reasoning_leak_escalates_without_retrying_primary(self):
        blocks = [
            {
                "seq": "1",
                "ts": "00:00:01,000 --> 00:00:02,000",
                "text": "Hallo",
            }
        ]
        primary = _FakeBackend(
            "gemini",
            "gemini-3.5-flash",
            ["1\n00:00:01,000 --> 00:00:02,000\nthinking: Hello there\n\n"],
        )
        escalation = _FakeBackend(
            "gemini",
            "gemini-3.1-pro-preview",
            ["1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"],
        )
        usage = make_usage_tracker()

        class Cfg:
            terminology = {}

        result = asyncio.run(
            translate_range(
                blocks,
                0,
                1,
                Cfg(),
                primary,
                escalation,
                usage,
                [0],
            )
        )

        self.assertEqual(1, primary.calls)
        self.assertEqual(1, escalation.calls)
        self.assertEqual(1, usage["escalation_count"])
        self.assertEqual(0, usage["retry_count"])
        self.assertEqual("Hello there.", result[0]["text"])

    def test_only_reasoning_like_structural_failures_escalate_immediately(self):
        self.assertTrue(
            _should_immediately_escalate_structural_failure(
                "Leaked reasoning or metadata text detected (found indicator 'thinking:')."
            )
        )
        self.assertTrue(
            _should_immediately_escalate_structural_failure(
                "Preamble detected (output does not start with a sequence number): 'Let me think...'"
            )
        )
        self.assertFalse(
            _should_immediately_escalate_structural_failure(
                "Block count mismatch: expected 100 blocks, got 99."
            )
        )

    def test_structural_failure_classification(self):
        self.assertEqual("reasoning_leak", _classify_structural_failure("Leaked reasoning or metadata text detected"))
        self.assertEqual("preamble", _classify_structural_failure("Preamble detected"))
        self.assertEqual("block_count", _classify_structural_failure("Block count mismatch: expected 100 blocks, got 99."))
        self.assertEqual("timestamp_count", _classify_structural_failure("Timestamp count mismatch: expected 2 '-->' indicators, got 1."))
        self.assertEqual("markup", _classify_structural_failure("HTML or subtitle markup detected; output plain SRT text only with real line breaks."))

    def test_block_count_retries_then_escalates(self):
        self.assertTrue(
            _should_retry_then_escalate_structural_failure(
                "Block count mismatch: expected 100 blocks, got 99."
            )
        )
        self.assertFalse(
            _should_retry_then_escalate_structural_failure(
                "Leaked reasoning or metadata text detected (found indicator 'thinking:')."
            )
        )

    def test_primary_attempt_limits_by_failure_class(self):
        self.assertEqual(
            1,
            _max_primary_attempts_for_structural_failure(
                "Leaked reasoning or metadata text detected (found indicator 'thinking:')."
            ),
        )
        self.assertEqual(
            2,
            _max_primary_attempts_for_structural_failure(
                "Block count mismatch: expected 100 blocks, got 99."
            ),
        )
        self.assertEqual(
            3,
            _max_primary_attempts_for_structural_failure(
                "Some other structural issue."
            ),
        )

    def test_mark_recommendations_prefers_clean_structural_winner(self):
        summary = {
            "candidates": [
                {
                    "candidate": "gemini:gemini-3.5-flash",
                    "structural_success": True,
                    "verifier_issue_count": 0,
                    "usage": {"retry_count": 0, "split_count": 0, "cost": 0.01},
                },
                {
                    "candidate": "gemini:gemini-3.1-pro-preview",
                    "structural_success": True,
                    "verifier_issue_count": 1,
                    "usage": {"retry_count": 0, "split_count": 0, "cost": 0.02},
                },
                {
                    "candidate": "openai:gpt-5.4",
                    "structural_success": False,
                    "verifier_issue_count": 0,
                    "usage": {"retry_count": 0, "split_count": 0, "cost": 0.03},
                },
            ]
        }
        _mark_recommendations(summary)
        winners = {c["candidate"]: (c["recommended_for_primary"], c["recommended_for_escalation"], c.get("rejected_reason", "")) for c in summary["candidates"]}
        self.assertEqual((True, True, ""), winners["gemini:gemini-3.5-flash"])
        self.assertEqual((False, False, "higher_verifier_issue_count"), winners["gemini:gemini-3.1-pro-preview"])
        self.assertEqual((False, False, "structural_failure"), winners["openai:gpt-5.4"])

    def test_block_count_retry_then_escalation_flow(self):
        blocks = [
            {
                "seq": "1",
                "ts": "00:00:01,000 --> 00:00:02,000",
                "text": "Hallo",
            }
        ]
        primary = _FakeBackend(
            "gemini",
            "gemini-3.5-flash",
            [
                "1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n2\n00:00:03,000 --> 00:00:04,000\nExtra.\n\n",
                "1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n2\n00:00:03,000 --> 00:00:04,000\nExtra.\n\n",
            ],
        )
        escalation = _FakeBackend(
            "gemini",
            "gemini-3.1-pro-preview",
            ["1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"],
        )
        usage = make_usage_tracker()
        chunk_state = {
            "preferred_backend": "",
            "last_failure_class": "",
            "primary_failed_structurally": False,
            "escalation_succeeded": False,
        }

        class Cfg:
            terminology = {}

        result = asyncio.run(
            translate_range(
                blocks,
                0,
                1,
                Cfg(),
                primary,
                escalation,
                usage,
                [0],
                chunk_state=chunk_state,
            )
        )

        self.assertEqual(2, primary.calls)
        self.assertEqual(1, escalation.calls)
        self.assertEqual("block_count", chunk_state["last_failure_class"])
        self.assertTrue(chunk_state["escalation_succeeded"])
        self.assertEqual("escalation", chunk_state["preferred_backend"])
        self.assertEqual("Hello there.", result[0]["text"])

    def test_chunk_memory_can_start_on_escalation(self):
        blocks = [
            {
                "seq": "1",
                "ts": "00:00:01,000 --> 00:00:02,000",
                "text": "Hallo",
            }
        ]
        primary = _FakeBackend("gemini", "gemini-3.5-flash", [])
        escalation = _FakeBackend(
            "gemini",
            "gemini-3.1-pro-preview",
            ["1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"],
        )
        usage = make_usage_tracker()
        chunk_state = {
            "preferred_backend": "escalation",
            "last_failure_class": "block_count",
            "primary_failed_structurally": True,
            "escalation_succeeded": True,
        }

        class Cfg:
            terminology = {}

        result = asyncio.run(
            translate_range(
                blocks,
                0,
                1,
                Cfg(),
                primary,
                escalation,
                usage,
                [0],
                chunk_state=chunk_state,
                use_escalation=True,
            )
        )

        self.assertEqual(0, primary.calls)
        self.assertEqual(1, escalation.calls)
        self.assertEqual("Hello there.", result[0]["text"])

    def test_failed_escalation_attempt_does_not_pin_chunk_to_escalation(self):
        blocks = [
            {
                "seq": "1",
                "ts": "00:00:01,000 --> 00:00:02,000",
                "text": "Hallo",
            }
        ]
        primary = _FakeBackend(
            "gemini",
            "gemini-3.5-flash",
            ["1\n00:00:01,000 --> 00:00:02,000\nthinking: Hello there\n\n"],
        )
        escalation = _FakeBackend(
            "gemini",
            "gemini-3.1-pro-preview",
            [
                "1\n00:00:01,000 --> 00:00:02,000\n<think>Hello there</think>\n\n",
                "1\n00:00:01,000 --> 00:00:02,000\n<think>Hello there</think>\n\n",
            ],
        )
        usage = make_usage_tracker()
        chunk_state = {
            "preferred_backend": "",
            "last_failure_class": "",
            "primary_failed_structurally": False,
            "escalation_succeeded": False,
        }

        class Cfg:
            terminology = {}

        result = asyncio.run(
            translate_range(
                blocks,
                0,
                1,
                Cfg(),
                primary,
                escalation,
                usage,
                [0],
                chunk_state=chunk_state,
            )
        )

        self.assertIsNone(result)
        self.assertEqual("", chunk_state["preferred_backend"])
        self.assertFalse(chunk_state["escalation_succeeded"])
        self.assertTrue(
            _should_start_chunk_on_escalation(
                {
                    "preferred_backend": "escalation",
                    "last_failure_class": "block_count",
                    "primary_failed_structurally": True,
                    "escalation_succeeded": True,
                },
                escalation,
            )
        )
        self.assertFalse(_should_start_chunk_on_escalation(chunk_state, escalation))

    def test_primary_success_clears_stale_escalation_memory(self):
        blocks = [
            {
                "seq": "1",
                "ts": "00:00:01,000 --> 00:00:02,000",
                "text": "Hallo",
            }
        ]
        primary = _FakeBackend(
            "gemini",
            "gemini-3.5-flash",
            ["1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"],
        )
        escalation = _FakeBackend("gemini", "gemini-3.1-pro-preview", [])
        usage = make_usage_tracker()
        chunk_state = {
            "preferred_backend": "escalation",
            "last_failure_class": "block_count",
            "primary_failed_structurally": True,
            "escalation_succeeded": True,
        }

        class Cfg:
            terminology = {}

        result = asyncio.run(
            translate_range(
                blocks,
                0,
                1,
                Cfg(),
                primary,
                escalation,
                usage,
                [0],
                chunk_state=chunk_state,
                use_escalation=False,
            )
        )

        self.assertEqual("Hello there.", result[0]["text"])
        self.assertEqual("primary", chunk_state["preferred_backend"])
        self.assertEqual("", chunk_state["last_failure_class"])
        self.assertFalse(chunk_state["primary_failed_structurally"])
        self.assertFalse(chunk_state["escalation_succeeded"])


if __name__ == "__main__":
    unittest.main()
