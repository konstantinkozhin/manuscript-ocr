__all__ = ["TRBA", "PPOCRv5Rec"]


def __getattr__(name):
    if name == "TRBA":
        from ._trba import TRBA

        return TRBA

    if name == "PPOCRv5Rec":
        from ._ppocrv5_rec import PPOCRv5Rec

        return PPOCRv5Rec

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
