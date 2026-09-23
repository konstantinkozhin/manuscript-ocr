__all__ = [
    "EAST",
    "Mask2Former",
    "YOLO",
]


def __getattr__(name):
    if name == "EAST":
        from ._east import EAST

        return EAST

    if name == "YOLO":
        from ._yolo import YOLO

        return YOLO

    if name == "Mask2Former":
        from ._mask2former import Mask2Former

        return Mask2Former

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
