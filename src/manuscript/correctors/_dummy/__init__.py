"""
Dummy text corrector implementation.

This module provides a simple corrector that returns a deep copy of the input
Page without any text modifications. Useful for testing pipeline integration.
"""

from typing import Optional, Union
from pathlib import Path

from manuscript.api.base import BaseModel
from manuscript.data import Page


class DummyCorrector(BaseModel):
    """
    A dummy corrector that returns a copy of the Page without modifications.

    This corrector is useful for:
    - Testing pipeline integration
    - Benchmarking baseline performance
    - Placeholder when no correction is needed

    The DummyCorrector creates a deep copy of the input Page, preserving
    all original text, coordinates, and confidence scores.

    Parameters
    ----------
    weights : str or Path, optional
        Not used for DummyCorrector, but accepted for API consistency.
    device : str, optional
        Not used for DummyCorrector, but accepted for API consistency.
    **kwargs
        Additional configuration options.

    Examples
    --------
    >>> from manuscript.correctors import DummyCorrector
    >>> from manuscript.data import Page, Block, Line, Word
    >>>
    >>> # Create a simple page
    >>> word = Word(
    ...     polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
    ...     detection_confidence=0.9,
    ...     text="hello"
    ... )
    >>> page = Page(blocks=[Block(lines=[Line(words=[word])])])
    >>>
    >>> # Apply dummy correction
    >>> corrector = DummyCorrector()
    >>> corrected = corrector.predict(page)
    >>>
    >>> # Text is preserved
    >>> print(corrected.blocks[0].lines[0].words[0].text)
    hello
    """

    # DummyCorrector doesn't need weights, but we set a default for API consistency
    default_weights_name = None
    pretrained_registry = {}

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the DummyCorrector.

        Parameters
        ----------
        weights : str or Path, optional
            Not used for DummyCorrector.
        device : str, optional
            Not used for DummyCorrector.
        **kwargs
            Additional configuration options.
        """
        # Skip BaseModel's weight resolution since we don't need weights
        self.device = device or "cpu"
        self.weights = None
        self.extra_config = kwargs
        self.session = None

    def _initialize_session(self):
        """Initialize session (no-op for DummyCorrector)."""
        pass

    def predict(self, page: Page) -> Page:
        """
        Return a deep copy of the input Page.

        Parameters
        ----------
        page : Page
            Input Page object with recognized text.

        Returns
        -------
        Page
            Deep copy of the input Page with all text preserved.

        Examples
        --------
        >>> corrector = DummyCorrector()
        >>> corrected_page = corrector.predict(page)
        """
        return page.model_copy(deep=True)
