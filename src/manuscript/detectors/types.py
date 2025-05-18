from pydantic import BaseModel, Field
from typing import List, Tuple


class Word(BaseModel):
    polygon: List[Tuple[float, float]] = Field(
        ..., description="List of vertices (x, y) of the polygon defining the region"
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
