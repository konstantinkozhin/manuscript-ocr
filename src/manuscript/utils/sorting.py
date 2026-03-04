from ..data import Page
from ..layouts import SimpleSorting


def organize_page(
    page: Page,
    max_splits: int = 10,
    use_columns: bool = True,
) -> Page:
    """
    Compatibility wrapper around ``SimpleSorting`` layout model.

    Parameters
    ----------
    page : Page
        Input page with detected words.
    max_splits : int, optional
        Maximum number of column split attempts. Default is 10.
    use_columns : bool, optional
        If True, segment into columns before line grouping. Default is True.

    Returns
    -------
    Page
        Organized page.
    """
    layout = SimpleSorting(max_splits=max_splits, use_columns=use_columns)
    return layout.predict(page)
