import unittest
from types import SimpleNamespace

from core.config import ShowConfig
from core.srt import parse_srt
from verify import check_srt, check_source_warnings, check_translations, evaluate_with_llm


def _cfg(**kwargs):
    base = {
        "name": "Test Show",
        "media_dir": "/tmp/media",
        "source_lang": "de",
        "target_lang": "en",
        "system_prompt": "Translate.",
    }
    base.update(kwargs)
    return ShowConfig(**base)


class VerifyHelperTests(unittest.TestCase):
    def test_evaluate_with_llm_uses_configured_judge_model(self):
        calls = {}

        class DummyModels:
            def generate_content(self, *, model, contents):
                calls["model"] = model
                calls["contents"] = contents
                return SimpleNamespace(text="PASS\nLooks good.")

        client = SimpleNamespace(models=DummyModels())
        cfg = _cfg(gemini_model="gemini-2.5-pro", llm_judge_model="judge-model-v1")
        blocks = parse_srt(
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo.\n\n"
        )

        verdict, reasoning = evaluate_with_llm(blocks, blocks, client, cfg)

        self.assertEqual("PASS", verdict)
        self.assertEqual("Looks good.", reasoning)
        self.assertEqual("judge-model-v1", calls["model"])

    def test_english_br_tag_reports_html_issue(self):
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hello<br/>there\n\n"
        )

        issues = check_srt(content, "en")

        self.assertTrue(any("HTML/subtitle markup" in issue for issue in issues))

    def test_english_bold_markers_still_report(self):
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "**Hello** there\n\n"
        )

        issues = check_srt(content, "en")

        self.assertTrue(any("Bold markers" in issue for issue in issues))

    def test_source_warning_reports_configured_message_on_block(self):
        cfg = _cfg(
            source_warnings=[
                {"pattern": r"\bBrühen\b", "message": "Likely Brillen."},
            ]
        )
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Die Brühen liegen da.\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "Alles gut.\n\n"
        )

        issues = check_source_warnings(parse_srt(content), cfg)

        self.assertIn("Block 1: source warning: Likely Brillen.", issues)

    def test_plain_red_hair_fragment_does_not_report_suspicious_literal(self):
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "red hair! Hooray!\n\n"
        )

        issues = check_srt(content, "en")

        self.assertFalse(any("red hair" in issue for issue in issues))

    def test_red_haired_man_phrase_does_not_report_anymore(self):
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "to the red-haired man!\n\n"
        )

        issues = check_srt(content, "en")

        self.assertFalse(any("red-haired man" in issue for issue in issues))

    def test_too_many_seas_phrase_does_not_report_anymore(self):
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "too many seas\n\n"
        )

        issues = check_srt(content, "en")

        self.assertFalse(any("too many seas" in issue for issue in issues))

    def test_check_translations_reports_mismatch_and_terminology_miss(self):
        cfg = _cfg(terminology={"Kobold": "kobold"})
        de_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Der Kobold ist da.\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "Noch ein Satz.\n\n"
        )
        en_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "The sprite is here.\n\n"
        )

        issues = check_translations(parse_srt(de_content), parse_srt(en_content), cfg)

        self.assertTrue(any("Block count mismatch" in issue for issue in issues))
        self.assertTrue(any("German term 'Kobold'" in issue for issue in issues))

    def test_check_translations_matches_terms_across_line_breaks(self):
        cfg = _cfg(terminology={"Herr Eder": "Herr Eder"})
        de_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Wiedersehen, Herr Eder.\n\n"
        )
        en_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Goodbye, Herr\n"
            "Eder.\n\n"
        )

        issues = check_translations(parse_srt(de_content), parse_srt(en_content), cfg)

        self.assertFalse(any("Herr Eder" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
