from pydantic import BaseModel, Field
from typing import List, Tuple


class Word(BaseModel):
    """
    Одна «словесная» область в документе.
    polygon: список точек (x, y), задающих вершины многоугольника, в порядке обхода.
    confidence: уверенность детектора.
    """
    polygon: List[Tuple[float, float]] = Field(
        ..., description="Список вершин (x, y) многоугольника, задающего область"
    )


class Block(BaseModel):
    """
    Блок текста, может состоять из нескольких слов (Word).
    """
    words: List[Word]


class Page(BaseModel):
    """
    Страница документа, содержит один или несколько текстовых блоков.
    """
    blocks: List[Block]