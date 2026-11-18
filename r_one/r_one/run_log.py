"""Optional per-frame log of perception results, used to score a run offline."""

import json


def track_record(track):
    """Return the run log entry for one tracker track."""
    x, y, z = track.position
    vx, vy = track.velocity
    return {
        'id': track.id,
        'class': track.class_name,
        'score': track.measurement.score,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'state': track.motion,
        'box': list(track.measurement.box),
        'size': list(track.measurement.size),
    }


def detection_record(detection, ranged):
    """Return the run log entry for one raw detection, and whether it got a position."""
    return {
        'class': detection.class_name,
        'score': float(detection.score),
        'box': [
            float(value) for value in (detection.x1, detection.y1, detection.x2, detection.y2)
        ],
        'ranged': bool(ranged),
    }


class RunLog:
    """Write one JSON line per processed frame. An empty path turns logging off."""

    def __init__(self, path: str = ''):
        # Line buffered, so a run stopped with Ctrl+C keeps every finished frame
        self._file = open(path, 'w', buffering=1, encoding='utf-8') if path else None

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def write_frame(self, stamp, latency_s, tracks, world_model, detections=(), counters=None):
        """
        Record one frame: its tracks and a snapshot of the static world model.

        detections holds detection_record entries. counters holds running totals, such as
        synced frames and TF failures, and each one is written as its own key.
        """
        if self._file is None:
            return
        record = {
            'stamp': stamp,
            'latency_s': latency_s,
            'tracks': tracks,
            'world_model': {
                name: list(position) for name, position in world_model.items()
            },
            'detections': list(detections),
        }
        record.update(counters or {})
        self._file.write(json.dumps(record, separators=(',', ':')) + '\n')

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
