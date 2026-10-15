__all__ = ["EAST", "YOLO", "RTDETR", "DocUFCN"]


def __getattr__(name):
    if name == "DocUFCN":
        from ._docufcn import DocUFCN

        return DocUFCN

    if name == "RTDETR":
        from ._rtdetr import RTDETR

        return RTDETR

    if name == "EAST":
        from ._east import EAST

        return EAST

    if name == "YOLO":
        from ._yolo import YOLO

        return YOLO

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
