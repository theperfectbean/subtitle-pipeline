import unittest

from core.pipeline import preflight_local_srt
from core.srt import (
    normalize_subtitle_text,
    parse_srt,
    validate_translation_structure,
    wrap_subtitle_text,
)


class SrtNormalizationTests(unittest.TestCase):
    def test_parse_srt_supports_unnumbered_blocks(self):
        raw = (
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo\n\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "Welt\n\n"
        )

        blocks = parse_srt(raw)

        self.assertEqual([b["seq"] for b in blocks], ["1", "2"])
        self.assertEqual(blocks[0]["ts"], "00:00:01,000 --> 00:00:02,000")
        self.assertEqual(blocks[1]["text"], "Welt")

    def test_br_tag_becomes_real_newline_before_wrapping(self):
        self.assertEqual(wrap_subtitle_text("Hello<br/>world"), "Hello\nworld")

    def test_font_tag_is_removed_preserving_text(self):
        self.assertEqual(
            normalize_subtitle_text('<font color="#ffff00">Hello</font>'),
            "Hello",
        )

    def test_spaces_before_punctuation_are_collapsed(self):
        self.assertEqual(
            normalize_subtitle_text("Hello , world !"),
            "Hello, world!",
        )


class TranslationValidationTests(unittest.TestCase):
    def test_preflight_local_srt_rejects_too_small_file(self):
        raw = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo\n\n"
        )

        count = preflight_local_srt("/tmp/sample.srt", raw, 100)

        self.assertEqual(0, count)

    def test_preflight_local_srt_uses_parsed_block_count(self):
        raw = (
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo\n\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "Welt\n\n"
        )

        count = preflight_local_srt("/tmp/sample.srt", raw, 2048)

        self.assertEqual(2, count)

    def test_validate_translation_structure_rejects_html_markup(self):
        source = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hallo\n\n"
        )
        raw = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hello<br/>there\n\n"
        )

        ok, message = validate_translation_structure(raw, parse_srt(source))

        self.assertFalse(ok)
        self.assertIn("HTML or subtitle markup detected", message)

    def test_valid_srt_with_matching_block_count_passes(self):
        raw = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "Hello there.\n\n"
        )

        ok, message = validate_translation_structure(raw, parse_srt(raw))

        self.assertTrue(ok)
        self.assertEqual(message, "")


if __name__ == "__main__":
    unittest.main()
