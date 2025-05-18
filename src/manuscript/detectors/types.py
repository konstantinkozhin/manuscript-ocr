from pydantic import BaseModel, Field
from typing import List, Tuple


class Word(BaseModel):
    polygon: List[Tuple[float, float]] = Field(
        ..., description="List of vertices (x, y) of the polygon defining the region"
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Score of the word detection between 0 and 1"
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
