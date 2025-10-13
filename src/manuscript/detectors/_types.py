from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class Word(BaseModel):
    polygon: List[Tuple[float, float]] = Field(
        ..., description="List of vertices (x, y) of the polygon defining the region"
    )
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Text detection confidence score from detector"
    )
    text: Optional[str] = Field(
        None, description="Recognized text content (populated by OCR pipeline)"
    )
    recognition_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Text recognition confidence score from recognizer"
    )


class Block(BaseModel):
    """
    A text block, which may consist of several words (Word).
    """

    words: List[Word]


class Page(BaseModel):
    """
    A document page containing one or multiple text blocks.
    """

    blocks: List[Block]
