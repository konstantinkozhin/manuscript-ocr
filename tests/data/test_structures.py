import json

import pytest
from pydantic import ValidationError

from manuscript.data import Block, Line, Page, TextSpan


class TestTextSpan:
    def test_text_span_creation_minimal(self):
        text_span = TextSpan(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
        )

        assert len(text_span.polygon) == 4
        assert text_span.detection_confidence == 0.95
        assert text_span.text is None
        assert text_span.recognition_confidence is None
        assert text_span.order is None

    def test_text_span_creation_full(self):
        text_span = TextSpan(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
            text="Hello",
            recognition_confidence=0.98,
            order=0,
        )

        assert text_span.text == "Hello"
        assert text_span.recognition_confidence == 0.98
        assert text_span.order == 0

    def test_text_span_accepts_polygon_with_more_than_four_points(self):
        text_span = TextSpan(
            polygon=[(0, 0), (5, 0), (10, 0), (10, 5), (5, 5), (0, 5)],
            detection_confidence=0.9,
        )

        assert len(text_span.polygon) == 6

    def test_text_span_requires_at_least_four_points(self):
        with pytest.raises(ValidationError):
            TextSpan(
                polygon=[(0, 0), (10, 0), (10, 5)],
                detection_confidence=0.9,
            )

    def test_text_span_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            TextSpan(
                polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
                detection_confidence=0.9,
                extra_field="unexpected",
            )


class TestLine:
    def test_line_uses_text_spans(self):
        text_span = TextSpan(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
            detection_confidence=0.9,
        )

        line = Line(text_spans=[text_span], order=3)

        assert len(line.text_spans) == 1
        assert line.text_spans[0] == text_span
        assert line.order == 3

    def test_line_rejects_legacy_words_key(self):
        with pytest.raises(ValidationError):
            Line.model_validate(
                {
                    "words": [
                        {
                            "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                            "detection_confidence": 0.9,
                        }
                    ]
                }
            )


class TestBlock:
    def test_block_with_lines(self):
        text_span = TextSpan(
            polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
            detection_confidence=0.9,
        )
        line = Line(text_spans=[text_span], order=0)

        block = Block(lines=[line], order=2)

        assert len(block.lines) == 1
        assert block.lines[0].text_spans[0] == text_span
        assert block.order == 2

    def test_block_wraps_flat_text_spans_into_single_line(self):
        text_spans = [
            TextSpan(
                polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
                detection_confidence=0.9,
            ),
            TextSpan(
                polygon=[(15, 0), (25, 0), (25, 5), (15, 5)],
                detection_confidence=0.95,
            ),
        ]

        block = Block(text_spans=text_spans)

        assert len(block.lines) == 1
        assert block.lines[0].text_spans == text_spans
        assert block.text_spans == text_spans

    def test_block_rejects_legacy_words_key(self):
        with pytest.raises(ValidationError):
            Block.model_validate(
                {
                    "words": [
                        {
                            "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                            "detection_confidence": 0.9,
                        }
                    ]
                }
            )


class TestPage:
    def test_page_complex_structure(self):
        line1 = Line(
            text_spans=[
                TextSpan(
                    polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
                    detection_confidence=0.9,
                    text="Hello",
                ),
                TextSpan(
                    polygon=[(15, 0), (30, 0), (30, 5), (15, 5)],
                    detection_confidence=0.95,
                    text="world",
                ),
            ],
            order=0,
        )
        line2 = Line(
            text_spans=[
                TextSpan(
                    polygon=[(0, 10), (10, 10), (10, 15), (0, 15)],
                    detection_confidence=0.92,
                    text="Again",
                )
            ],
            order=1,
        )
        page = Page(blocks=[Block(lines=[line1, line2], order=0)])

        assert len(page.blocks) == 1
        assert len(page.blocks[0].lines) == 2
        assert page.blocks[0].lines[0].text_spans[0].text == "Hello"
        assert page.blocks[0].lines[0].text_spans[1].text == "world"
        assert page.blocks[0].lines[1].text_spans[0].text == "Again"

    def test_page_serialization_roundtrip(self):
        page = Page(
            blocks=[
                Block(
                    lines=[
                        Line(
                            text_spans=[
                                TextSpan(
                                    polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
                                    detection_confidence=0.9,
                                    text="Test",
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        dumped = page.model_dump()
        restored = Page.model_validate(dumped)

        assert dumped["blocks"][0]["lines"][0]["text_spans"][0]["text"] == "Test"
        assert restored.blocks[0].lines[0].text_spans[0].text == "Test"

    def test_page_rejects_legacy_words_json(self):
        legacy_payload = {
            "blocks": [
                {
                    "lines": [
                        {
                            "words": [
                                {
                                    "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                                    "detection_confidence": 0.9,
                                    "text": "Hello",
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        with pytest.raises(ValidationError):
            Page.model_validate(legacy_payload)

    def test_page_to_json_and_from_json(self, tmp_path):
        page = Page(
            blocks=[
                Block(
                    lines=[
                        Line(
                            text_spans=[
                                TextSpan(
                                    polygon=[(0, 0), (10, 0), (10, 5), (0, 5)],
                                    detection_confidence=0.9,
                                    text="Saved",
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        json_path = tmp_path / "page.json"
        json_text = page.to_json(json_path)
        restored = Page.from_json(json_path)

        assert json.loads(json_text)["blocks"][0]["lines"][0]["text_spans"][0]["text"] == "Saved"
        assert restored.blocks[0].lines[0].text_spans[0].text == "Saved"
