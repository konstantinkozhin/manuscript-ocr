from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Union
from pathlib import Path
import json


class Word(BaseModel):
    """
    A single detected or recognized word.

    Attributes
    ----------
    polygon : List[Tuple[float, float]]
        Polygon vertices (x, y), ordered clockwise. For quadrilateral text regions:
        TL → TR → BR → BL (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
    detection_confidence : float
        Text detection confidence score from detector (0.0 to 1.0).
    text : str, optional
        Recognized text content (populated by OCR pipeline). None if only detection
        was performed.
    recognition_confidence : float, optional
        Text recognition confidence score from recognizer (0.0 to 1.0). None if only
        detection was performed.
    order : int, optional
        Word position inside the line after sorting. None before sorting.

    Examples
    --------
    >>> word = Word(
    ...     polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
    ...     detection_confidence=0.95,
    ...     text="Hello",
    ...     recognition_confidence=0.98
    ... )
    >>> print(word.text)
    Hello
    """

    polygon: List[Tuple[float, float]] = Field(
        ...,
        min_length=4,
        description="Polygon vertices (x, y), ordered clockwise. For quadrilateral text regions: TL → TR → BR → BL.",
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
        description="Word position inside the line after sorting. None before sorting.",
    )


class Line(BaseModel):
    """
    A single text line containing one or more words.

    Attributes
    ----------
    words : List[Word]
        List of words in the line.
    order : int, optional
        Line position inside a block or page after sorting. None before sorting.

    Examples
    --------
    >>> line = Line(words=[
    ...     Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
    ...          detection_confidence=0.95, text="Hello"),
    ...     Word(polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
    ...          detection_confidence=0.97, text="World"),
    ... ])
    >>> print(len(line.words))
    2
    """

    words: List[Word]
    order: Optional[int] = Field(
        None,
        description="Line position inside a block or page after sorting. None before sorting.",
    )


class Block(BaseModel):
    """
    A logical text block (e.g., paragraph, column).

    Attributes
    ----------
    lines : List[Line]
        List of text lines in the block.
    words : List[Word], optional
        Legacy: Direct list of words without line structure. Used for backward
        compatibility. If both `lines` and `words` are empty, creates a single
        line from words.
    order : int, optional
        Block reading-order position after sorting. None before sorting.

    Examples
    --------
    >>> block = Block(lines=[
    ...     Line(words=[Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
    ...                      detection_confidence=0.95, text="Line 1")]),
    ...     Line(words=[Word(polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
    ...                      detection_confidence=0.97, text="Line 2")]),
    ... ])
    >>> print(len(block.lines))
    2
    """

    lines: List[Line] = Field(default_factory=list)
    words: List[Word] = Field(
        default_factory=list,
        description="Legacy: Direct list of words. Use 'lines' for structured output.",
    )
    order: Optional[int] = Field(
        None,
        description="Block reading-order position after sorting. None before sorting.",
    )

    def __init__(self, **data):
        """Initialize Block with backward compatibility for words-only input."""
        super().__init__(**data)
        # If lines is empty but words is provided, wrap words in a single line
        if not self.lines and self.words:
            self.lines = [Line(words=self.words)]


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
    ...         Line(words=[Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
    ...                          detection_confidence=0.95, text="Hello")])
    ...     ])
    ... ])
    >>> print(len(page.blocks))
    1
    """

    blocks: List[Block]

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """
        Export Page to JSON.

        Parameters
        ----------
        path : str or Path, optional
            If provided, saves JSON to file.
        indent : int, optional
            JSON indentation. Default is 2.

        Returns
        -------
        str
            JSON string representation.

        Examples
        --------
        >>> page.to_json("result.json")  # save to file
        >>> json_str = page.to_json()    # get as string
        """
        data = self.model_dump()
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
