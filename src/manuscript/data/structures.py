import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SCHEMA_CURRENT = "v0_1_11"
_SCHEMA_LEGACY = "v0_1_10"
_SUPPORTED_SCHEMAS = {_SCHEMA_CURRENT, _SCHEMA_LEGACY}


class TextSpan(BaseModel):
    """
    A single detected or recognized text span.

    A text span is the smallest OCR region in the pipeline. It may correspond
    to a word, a whole text line, or any other contiguous text segment returned
    by a detector.

    Attributes
    ----------
    polygon : List[Tuple[float, float]]
        Polygon vertices (x, y), ordered clockwise.
        The public data model supports arbitrary polygons with 4 or more points.
        For quadrilateral text regions, the canonical order is
        TL -> TR -> BR -> BL (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
    detection_confidence : float
        Text detection confidence score from detector (0.0 to 1.0).
    text : str, optional
        Recognized text content (populated by OCR pipeline). None if only detection
        was performed.
    recognition_confidence : float, optional
        Text recognition confidence score from recognizer (0.0 to 1.0). None if only
        detection was performed.
    order : int, optional
        Text span position inside the line after sorting. None before sorting.

    Examples
    --------
    >>> text_span = TextSpan(
    ...     polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
    ...     detection_confidence=0.95,
    ...     text="Hello",
    ...     recognition_confidence=0.98
    ... )
    >>> print(text_span.text)
    Hello
    """

    model_config = ConfigDict(extra="forbid")

    polygon: List[Tuple[float, float]] = Field(
        ...,
        min_length=4,
        description=(
            "Polygon vertices (x, y), ordered clockwise. Supports arbitrary "
            "polygons with 4 or more points. For quadrilateral text regions: "
            "TL -> TR -> BR -> BL."
        ),
    )
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Text detection confidence score from detector"
    )
    text: Optional[str] = Field(
        None, description="Recognized text content (populated by OCR pipeline)"
    )
    recognition_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Text recognition confidence score from recognizer",
    )
    order: Optional[int] = Field(
        None,
        description="Text span position inside the line after sorting. None before sorting.",
    )


def __getattr__(name: str):
    if name == "Word":
        return TextSpan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Line(BaseModel):
    """
    A single text line containing one or more text spans.

    Attributes
    ----------
    text_spans : List[TextSpan]
        List of text spans in the line.
    order : int, optional
        Line position inside a block or page after sorting. None before sorting.

    Examples
    --------
    >>> line = Line(text_spans=[
    ...     TextSpan(
    ...         polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
    ...         detection_confidence=0.95,
    ...         text="Hello",
    ...     ),
    ...     TextSpan(
    ...         polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
    ...         detection_confidence=0.97,
    ...         text="World",
    ...     ),
    ... ])
    >>> print(len(line.text_spans))
    2
    """

    model_config = ConfigDict(extra="forbid")

    text_spans: List[TextSpan] = Field(
        default_factory=list,
        description="List of text spans in the line.",
    )
    order: Optional[int] = Field(
        None,
        description="Line position inside a block or page after sorting. None before sorting.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_words_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "words" not in data:
            return data

        normalized = dict(data)
        legacy_words = normalized.pop("words")

        if "text_spans" in normalized and normalized["text_spans"] != legacy_words:
            raise ValueError(
                "Line received both 'words' and 'text_spans' with different values."
            )

        normalized.setdefault("text_spans", legacy_words)
        return normalized

    @property
    def words(self) -> List[TextSpan]:
        """Backward-compatible alias for ``text_spans``."""
        return self.text_spans

    @words.setter
    def words(self, value: List[TextSpan]) -> None:
        self.text_spans = value


class Block(BaseModel):
    """
    A logical text block (e.g., paragraph, column).

    Attributes
    ----------
    lines : List[Line]
        List of text lines in the block.
    text_spans : List[TextSpan], optional
        Optional flat list of text spans used as a shorthand input. If ``lines``
        is empty and ``text_spans`` are provided, they are wrapped into a
        single line.
    order : int, optional
        Block reading-order position after sorting. None before sorting.

    Examples
    --------
    >>> block = Block(lines=[
    ...     Line(text_spans=[
    ...         TextSpan(
    ...             polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
    ...             detection_confidence=0.95,
    ...             text="Line 1",
    ...         )
    ...     ]),
    ...     Line(text_spans=[
    ...         TextSpan(
    ...             polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
    ...             detection_confidence=0.97,
    ...             text="Line 2",
    ...         )
    ...     ]),
    ... ])
    >>> print(len(block.lines))
    2
    """

    model_config = ConfigDict(extra="forbid")

    lines: List[Line] = Field(default_factory=list)
    text_spans: List[TextSpan] = Field(
        default_factory=list,
        description=(
            "Optional flat list of text spans. Use 'lines' for structured output."
        ),
    )
    order: Optional[int] = Field(
        None,
        description="Block reading-order position after sorting. None before sorting.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_words_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "words" not in data:
            return data

        normalized = dict(data)
        legacy_words = normalized.pop("words")

        if "text_spans" in normalized and normalized["text_spans"] != legacy_words:
            raise ValueError(
                "Block received both 'words' and 'text_spans' with different values."
            )

        normalized.setdefault("text_spans", legacy_words)
        return normalized

    def __init__(self, **data):
        """Initialize Block, normalizing flat ``text_spans`` into one line."""
        super().__init__(**data)
        if not self.lines and self.text_spans:
            self.lines = [Line(text_spans=self.text_spans)]

    @property
    def words(self) -> List[TextSpan]:
        """Backward-compatible alias for flat ``text_spans`` input."""
        return self.text_spans

    @words.setter
    def words(self, value: List[TextSpan]) -> None:
        self.text_spans = value


class Page(BaseModel):
    """
    A document page containing blocks of text.

    For a full visual diagram of the data model, see:
    ``DATA_MODEL.md`` located in the same module directory.

    Attributes
    ----------
    blocks : List[Block]
        List of text blocks on the page.

    Examples
    --------
    >>> page = Page(blocks=[
    ...     Block(lines=[
    ...         Line(text_spans=[
    ...             TextSpan(
    ...                 polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
    ...                 detection_confidence=0.95,
    ...                 text="Hello",
    ...             )
    ...         ])
    ...     ])
    ... ])
    >>> print(len(page.blocks))
    1
    """

    model_config = ConfigDict(extra="forbid")

    blocks: List[Block]

    @staticmethod
    def _normalize_schema(schema: str) -> str:
        if schema not in _SUPPORTED_SCHEMAS:
            supported = ", ".join(sorted(_SUPPORTED_SCHEMAS))
            raise ValueError(f"schema must be one of {{{supported}}}, got: {schema}")
        return schema

    @staticmethod
    def _span_to_legacy_dict(span: TextSpan) -> Dict[str, Any]:
        return span.model_dump()

    @classmethod
    def _line_to_dict(cls, line: Line, schema: str) -> Dict[str, Any]:
        if schema == _SCHEMA_CURRENT:
            return line.model_dump()

        return {
            "words": [cls._span_to_legacy_dict(span) for span in line.text_spans],
            "order": line.order,
        }

    @classmethod
    def _block_to_dict(cls, block: Block, schema: str) -> Dict[str, Any]:
        if schema == _SCHEMA_CURRENT:
            return block.model_dump()

        return {
            "lines": [cls._line_to_dict(line, schema) for line in block.lines],
            "words": [cls._span_to_legacy_dict(span) for span in block.text_spans],
            "order": block.order,
        }

    def to_dict(self, schema: str = _SCHEMA_CURRENT) -> Dict[str, Any]:
        """
        Export Page to a plain Python dictionary.

        Parameters
        ----------
        schema : {"v0_1_11", "v0_1_10"}, optional
            Output schema version. Default is ``"v0_1_11"``.
        """
        schema = self._normalize_schema(schema)
        if schema == _SCHEMA_CURRENT:
            return self.model_dump()

        return {
            "blocks": [self._block_to_dict(block, schema) for block in self.blocks]
        }

    def to_json(
        self,
        path: Optional[Union[str, Path]] = None,
        indent: int = 2,
        schema: str = _SCHEMA_CURRENT,
    ) -> str:
        """
        Export Page to JSON.

        Parameters
        ----------
        path : str or Path, optional
            If provided, saves JSON to file.
        indent : int, optional
            JSON indentation. Default is 2.
        schema : {"v0_1_11", "v0_1_10"}, optional
            Output schema version. Default is ``"v0_1_11"``.

        Returns
        -------
        str
            JSON string representation.

        Examples
        --------
        >>> page.to_json("result.json")  # save to file
        >>> json_str = page.to_json()    # get as string
        >>> legacy_json = page.to_json(schema="v0_1_10")
        """
        data = self.to_dict(schema=schema)
        json_str = json.dumps(data, ensure_ascii=False, indent=indent)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "Page":
        """
        Load Page from JSON file or string.

        Parameters
        ----------
        source : str or Path
            Path to JSON file or JSON string.

        Returns
        -------
        Page
            Loaded Page object.

        Examples
        --------
        >>> page = Page.from_json("result.json")
        >>> page = Page.from_json('{"blocks": [...]}')
        """
        path = Path(source)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(source)
        return cls.model_validate(data)
