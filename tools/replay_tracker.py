#!/usr/bin/env python3
"""
Replay a run log's raw positions through the tracker and write a new run log.

The input's x, y and z must be measured positions, as in the recorded fixture. Run:
    python3 tools/replay_tracker.py r_one/test/fixtures/sample_run_baseline.jsonl out.jsonl
"""

import argparse
import sys

from r_one.run_log import RunLog, track_record
from r_one.tracking import Measurement, Tracker
from r_one.world_model import WorldModel
from score_run import read_frames


def frame_measurements(frame):
    """Turn one logged frame's tracks back into measurements."""
    return [
        Measurement(
            track['x'], track['y'], track['z'], track['class'], track['score'],
            tuple(track.get('box', ())),
        )
        for track in frame['tracks']
    ]


def replay(frames, tracker=None, world_model=None):
    """Run the tracker over the frames and return the frames it would have logged."""
    tracker = tracker or Tracker()
    world_model = world_model if world_model is not None else WorldModel()
    output = []
    for frame in frames:
        stamp = frame['stamp']
        tracks = tracker.step(stamp, frame_measurements(frame))
        for track in tracks:
            world_model.observe(track, stamp)
        output.append({
            'stamp': stamp,
            'latency_s': frame['latency_s'],
            'tracks': [track_record(track) for track in tracks],
            'world_model': {name: list(position) for name, position in world_model.items()},
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('run_log', help='Run log whose positions are raw measurements')
    parser.add_argument('out', help='Run log to write')
    args = parser.parse_args(argv)

    frames = read_frames(args.run_log)
    if not frames:
        print(f'No frames in {args.run_log}', file=sys.stderr)
        return 2

    log = RunLog(args.out)
    for frame in replay(frames):
        log.write_frame(
            frame['stamp'], frame['latency_s'], frame['tracks'], frame['world_model']
        )
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
