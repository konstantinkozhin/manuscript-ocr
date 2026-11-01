__all__ = ["EAST", "YOLO", "ConsensusWBFDetector"]


def __getattr__(name):
    if name == "ConsensusWBFDetector":
        from ._consensus import ConsensusWBFDetector

        return ConsensusWBFDetector

    if name == "EAST":
        from ._east import EAST

        return EAST

    if name == "YOLO":
        from ._yolo import YOLO

        return YOLO

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
