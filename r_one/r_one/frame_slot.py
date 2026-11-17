"""A one-frame hand-off between the sensor callbacks and the perception worker."""

import threading
import time


class FrameSlot:
    """
    Hold only the newest frame, so the worker never processes a stale one.

    put() never blocks, so a slow detector cannot hold up the callbacks that feed it.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, frame) -> bool:
        """Store frame, and return True if it replaced one that was never taken."""
        with self._condition:
            replaced = self._frame is not None
            self._frame = frame
            self._condition.notify()
            return replaced

    def take(self, timeout: float):
        """Return the newest frame, or None after timeout seconds or once closed."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            frame, self._frame = self._frame, None
            return frame

    def close(self):
        """Wake any waiting take(). Every later take() returns None."""
        with self._condition:
            self._closed = True
            self._frame = None
            self._condition.notify_all()
