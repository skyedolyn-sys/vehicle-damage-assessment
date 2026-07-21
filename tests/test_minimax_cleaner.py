"""Unit tests for clean_minimax_output and JSON extraction helpers.

These tests cover the heuristic that scans LLM output for JSON, even when:
- JSON is hidden inside _CONTENT_tag blocks
- JSON is embedded inside narrative prose
- narrative contains fake `{}` or string-literal braces
- there is no JSON at all (expect fallback to cleaned prose)
"""

import json

import pytest

from agents.minimax_client import (
    _find_first_valid_json,
    _split_think_tags,
    clean_minimax_output,
)


THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"


class TestCleanMinimaxOutput:
    """End-to-end behaviour of clean_minimax_output."""

    def test_pure_json_input(self):
        """A clean JSON object passes through unchanged."""
        raw = '{"a": 1, "b": [2, 3], "c": {"d": "x"}}'
        out = clean_minimax_output(raw)
        assert json.loads(out) == {"a": 1, "b": [2, 3], "c": {"d": "x"}}

    def test_json_outside_think_tags(self):
        """JSON appearing after a closing 标记 gets extracted."""
        raw = (
            THINK_OPEN
            + "I am thinking about the car damage carefully. Many thoughts..."
            + THINK_CLOSE
            + ' Final answer: {"result": "ok", "score": 0.9}'
        )
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed == {"result": "ok", "score": 0.9}

    def test_json_inside_think_tags(self):
        """When the only JSON lives inside 标记块, the new cleaner recovers it."""
        raw = (
            THINK_OPEN
            + 'Here is the JSON response: {"inside": true, "n": 7} and more notes'
            + THINK_CLOSE
        )
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed == {"inside": True, "n": 7}

    def test_json_embedded_in_narrative_no_think_tags(self):
        """Narrative with an embedded JSON object (no think markup at all)."""
        raw = (
            "I analyzed the photo. Lots of prose about the bumper.\n"
            "Now the answer: {\"severity\": \"minor\", \"parts\": [\"bumper\"]}\n"
            "End of narrative."
        )
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed["severity"] == "minor"
        assert parsed["parts"] == ["bumper"]

    def test_fake_empty_braces_in_narrative_are_ignored(self):
        """If only empty {} strings exist, the cleaned text is returned for fallback
        (which is fine because `json.loads("{}")` returns {} anyway and the caller
        decides what to do with it). The key invariant: we do NOT crash and we
        produce a parseable string."""
        raw = "Here is {} some prose {x} about nothing {} structured {}"
        out = clean_minimax_output(raw)
        assert out.strip() != ""
        # Either the empty `{}` is returned (valid JSON dict) or the cleaned narrative.
        # Both are acceptable; the cleaner must NOT raise.
        try:
            json.loads(out)
        except json.JSONDecodeError:
            # If we did return the narrative, it must still contain the prose.
            assert "structured" in out

    def test_multiple_json_blocks_returns_first_valid(self):
        """If multiple JSON objects exist, return the earliest parseable one."""
        raw = (
            "garbage {not-json}\n"
            '{"first": true, "id": 1}\n'
            "more prose\n"
            '{"second": true, "id": 2}\n'
        )
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed["first"] is True
        assert parsed["id"] == 1

    def test_nested_json_object(self):
        """Nested JSON is found via brace-stack matching."""
        raw = 'narrative prefix {"a": {"b": {"c": [1, 2, 3]}}} suffix'
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed == {"a": {"b": {"c": [1, 2, 3]}}}

    def test_top_level_list_json(self):
        """Top-level arrays are also valid JSON targets."""
        raw = 'list payload [{"x": 1}, {"x": 2}, {"x": 3}] trailing prose'
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[-1]["x"] == 3

    def test_braces_inside_string_values_are_ignored(self):
        """A JSON string containing literal `{...}` should not trap the matcher."""
        raw = (
            'Note: the message reads "hello {world}" verbatim.\n'
            '{"decision": "extract", "ok": true}'
        )
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed == {"decision": "extract", "ok": True}
        # And the embedded `{}` snippet from the prose must NOT be what we return.
        assert out.startswith("{") is False or '"decision"' in out

    def test_no_json_anywhere_returns_narrative_fallback(self):
        """Pure narrative without any JSON objects/arrays returns the cleaned
        prose (the caller's fallback path picks it up from there)."""
        raw = (
            "I observed the photo carefully. The car appears damaged in many areas.\n"
            "There is evidence of frontal impact, scrapes, and dents.\n"
            "No structured output follows this paragraph. Total narrative only."
        )
        out = clean_minimax_output(raw)
        # The cleaned narrative should still mention key words.
        assert "damaged" in out
        assert out.strip() != ""
        # Crucially, no JSON dict/list was hidden in the prose. We confirm this
        # by re-running the dedicated finder, which returns "".
        assert _find_first_valid_json(out) == ""

    def test_strips_markdown_fences_before_searching(self):
        """Markdown code fences are stripped first, then JSON is extracted."""
        raw = (
            "Some prefix\n"
            "```json\n"
            '{"fenced": true, "value": 42}\n'
            "```\n"
            "trailing chatter"
        )
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed == {"fenced": True, "value": 42}

    def test_unclosed_think_tag_still_searches(self):
        """If 标记 is opened but never closed, the entire body is treated as inside."""
        raw = THINK_OPEN + 'thinking but no close {"found": "yes", "n": 11}'
        out = clean_minimax_output(raw)
        parsed = json.loads(out)
        assert parsed == {"found": "yes", "n": 11}


class TestSplitThinkTags:
    """Direct tests for the tag-splitting helper."""

    def test_no_tags_returns_whole_text_outside(self):
        outside, inside = _split_think_tags("plain text")
        assert "".join(outside) == "plain text"
        assert "".join(inside) == ""

    def test_text_only_inside(self):
        outside, inside = _split_think_tags(THINK_OPEN + "thoughts" + THINK_CLOSE)
        assert "".join(outside) == ""
        assert "".join(inside) == "thoughts"

    def test_interleaved(self):
        raw = "A" + THINK_OPEN + "B" + THINK_CLOSE + "C" + THINK_OPEN + "D" + THINK_CLOSE + "E"
        outside, inside = _split_think_tags(raw)
        assert "".join(outside) == "ACE"
        assert "".join(inside) == "BD"

    def test_unclosed_tag(self):
        raw = "before " + THINK_OPEN + "after no close"
        outside, inside = _split_think_tags(raw)
        assert "".join(outside) == "before "
        assert "".join(inside) == "after no close"


class TestFindFirstValidJson:
    """Direct tests for the stack-matching JSON finder."""

    def test_returns_empty_for_empty_input(self):
        assert _find_first_valid_json("") == ""
        assert _find_first_valid_json(None) == ""

    def test_finds_simple_object(self):
        txt = 'prose {"a": 1, "b": 2} prose'
        out = _find_first_valid_json(txt)
        assert json.loads(out) == {"a": 1, "b": 2}

    def test_skips_unbalanced_braces(self):
        txt = 'prose {not-json} correct {"ok": true} done'
        out = _find_first_valid_json(txt)
        assert json.loads(out) == {"ok": True}

    def test_top_level_array(self):
        txt = 'intro [1, 2, 3] outro'
        out = _find_first_valid_json(txt)
        assert json.loads(out) == [1, 2, 3]

    def test_string_with_braces_is_safe(self):
        """Braces inside a string literal must not start a match attempt."""
        txt = '"hello {world}" and {"real": "json"}'
        out = _find_first_valid_json(txt)
        # Should land on the second, balanced JSON object.
        assert json.loads(out) == {"real": "json"}

    def test_quick_path_whole_object(self):
        """Fast path: whole text is JSON."""
        txt = '{"fast": "path"}'
        assert _find_first_valid_json(txt) == txt

    def test_returns_empty_when_no_json(self):
        txt = "no braces here at all"
        assert _find_first_valid_json(txt) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
