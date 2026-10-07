__all__ = ["TRBA", "PPOCRRec", "TrOCR"]


def __getattr__(name):
    if name == "TrOCR":
        from ._trocr import TrOCR

        return TrOCR

    if name == "PPOCRRec":
        from ._ppocr_rec import PPOCRRec

        return PPOCRRec

    if name == "TRBA":
        from ._trba import TRBA

        return TRBA

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
