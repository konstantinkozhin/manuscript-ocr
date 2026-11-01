__all__ = [
    "CharLM",
]


def __getattr__(name):
    if name == "CharLM":
        from ._charlm import CharLM

        return CharLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
