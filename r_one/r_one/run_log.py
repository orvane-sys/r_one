"""Optional per-frame log of perception results, used to score a run offline."""

import json


class RunLog:
    """Write one JSON line per processed frame. An empty path turns logging off."""

    def __init__(self, path: str = ''):
        # Line buffered, so a run stopped with Ctrl+C keeps every finished frame
        self._file = open(path, 'w', buffering=1, encoding='utf-8') if path else None

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def write_frame(self, stamp, latency_s, tracks, world_model):
        """Record one frame: its tracks and a snapshot of the static world model."""
        if self._file is None:
            return
        record = {
            'stamp': stamp,
            'latency_s': latency_s,
            'tracks': tracks,
            'world_model': {
                name: list(position) for name, position in world_model.items()
            },
        }
        self._file.write(json.dumps(record, separators=(',', ':')) + '\n')

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
