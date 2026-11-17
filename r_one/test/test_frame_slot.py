"""Tests for the one-frame hand-off between the sensor callbacks and the worker."""

import threading
import time
import unittest

from r_one.frame_slot import FrameSlot


class TestFrameSlot(unittest.TestCase):
    def test_the_newest_frame_wins(self):
        slot = FrameSlot()
        slot.put('first')
        slot.put('second')
        self.assertEqual(slot.take(timeout=0.1), 'second')
        self.assertIsNone(slot.take(timeout=0.0))

    def test_put_reports_a_replaced_frame(self):
        slot = FrameSlot()
        self.assertFalse(slot.put('first'))
        self.assertTrue(slot.put('second'))
        slot.take(timeout=0.1)
        self.assertFalse(slot.put('third'))

    def test_take_times_out_when_empty(self):
        slot = FrameSlot()
        start = time.monotonic()
        self.assertIsNone(slot.take(timeout=0.05))
        self.assertGreaterEqual(time.monotonic() - start, 0.05)

    def test_take_wakes_when_a_frame_arrives(self):
        slot = FrameSlot()
        threading.Timer(0.05, slot.put, args=('frame',)).start()
        self.assertEqual(slot.take(timeout=5.0), 'frame')

    def test_close_wakes_a_waiting_take(self):
        slot = FrameSlot()
        result = []
        waiter = threading.Thread(target=lambda: result.append(slot.take(timeout=5.0)))
        waiter.start()
        time.sleep(0.05)
        slot.close()
        waiter.join(timeout=1.0)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [None])
        self.assertTrue(slot.closed)

    def test_take_after_close_returns_none(self):
        slot = FrameSlot()
        slot.put('frame')
        slot.close()
        self.assertIsNone(slot.take(timeout=0.1))


if __name__ == '__main__':
    unittest.main()
