"""What every detector returns, so the node does not depend on one detection library."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One box in image pixels. track_id is -1 when the detector gives no ID."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    score: float
    track_id: int = -1
