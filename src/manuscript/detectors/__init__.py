__all__ = [
    "EAST",
    "EASTV2",
    "YOLO",
]


def __getattr__(name):
    if name == "EAST":
        from ._east import EAST

        return EAST

    if name == "EASTV2":
        from ._eastv2 import EASTV2

        return EASTV2

    if name == "YOLO":
        from ._yolo import YOLO

        return YOLO

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
