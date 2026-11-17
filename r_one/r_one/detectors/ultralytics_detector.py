"""Ultralytics YOLO in plain detection mode, returning plain Detection objects."""

import os
from typing import List

from r_one.detectors.base import Detection

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False


def resolve_model_path(name: str) -> str:
    """Prefer a weights file in $R_ONE_MODEL_DIR (set in the Docker image) over a download."""
    model_dir = os.environ.get('R_ONE_MODEL_DIR')
    if model_dir and not os.path.isabs(name):
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return name


class UltralyticsDetector:
    """
    Detect with one YOLO model. Tracking is left to r_one.tracking.

    predict() is used, not track(): ByteTrack withholds the box of a newly seen object
    for its first frame, which loses a person who reappears after a gap.
    """

    def __init__(self, model_path: str, img_size: int):
        self.model = YOLO(model_path)
        self.img_size = img_size
        self._last_results = None

    def detect(self, image) -> List[Detection]:
        results = self.model.predict(image, imgsz=self.img_size, verbose=False)
        self._last_results = results
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            detections.append(Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                class_name=self.model.names[int(box.cls[0])],
                score=float(box.conf[0]),
                track_id=int(box.id[0]) if box.id is not None else -1,
            ))
        return detections

    def plot(self):
        """Return the last frame with its boxes drawn, or None before the first frame."""
        if self._last_results is None:
            return None
        return self._last_results[0].plot()
