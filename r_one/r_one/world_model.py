"""Remember where static objects are, after their tracks have ended."""

from collections import OrderedDict
from typing import Dict, Iterator, Tuple

from r_one.tracking import DYNAMIC, STATIC, Track

Position = Tuple[float, float, float]


class WorldModel:
    """Positions of proven-static objects, capped at max_entries."""

    def __init__(self, max_entries: int = 256, min_static_age_s: float = 1.0):
        self.max_entries = max_entries
        self.min_static_age_s = min_static_age_s
        # Least recently seen first, so the oldest entry is evicted first.
        self._entries: 'OrderedDict[str, Position]' = OrderedDict()

    @staticmethod
    def key(track: Track) -> str:
        return f'{track.class_name}_{track.id}'

    def observe(self, track: Track, t: float):
        """Store a confirmed track that has proven static, and forget one that moves."""
        key = self.key(track)
        if track.motion == DYNAMIC:
            self._entries.pop(key, None)
        elif track.motion == STATIC and t - track.created >= self.min_static_age_s:
            self._entries[key] = track.position
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def items(self) -> Iterator[Tuple[str, Position]]:
        return iter(list(self._entries.items()))

    def as_dict(self) -> Dict[str, Position]:
        return dict(self._entries)

    def clear(self):
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
