__all__ = [
    "EAST",
    "YOLO",
]


def __getattr__(name):
    if name == "EAST":
        from ._east import EAST

        return EAST

    if name == "YOLO":
        from ._yolo import YOLO

        return YOLO

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
