#!/usr/bin/env python3
"""Score a perception run log against the demo world's ground truth.

Produce a log with the run_log_path parameter, then run:
    python3 tools/score_run.py reports/run.jsonl --gates
"""

import argparse
import json
import math
from pathlib import Path
import sys

from ground_truth import load_actor_paths

DEFAULT_WORLD = (
    Path(__file__).resolve().parent.parent / 'r_one_bringup' / 'worlds' / 'spatial_demo.world'
)
WALKER = 'walking_human'
STANDING = 'standing_person'
PERSON_CLASS = 'person'

# A detection further than this from every actor is not matched to any of them.
MATCH_RADIUS_M = 1.0
# Used to keep the standing person out of the walker's time fit, and to count ghosts.
STATIC_RADIUS_M = 0.75
ALIGNMENT_RADIUS_M = 0.5
OFFSET_STEP_S = 0.05
# A larger fitted shift is reported, because the actors should start with sim time.
EXPECTED_OFFSET_S = 0.5
# Velocity right after a turn is excluded from the cruise figure.
TURN_SETTLE_S = 1.0
ID_SWITCH_WINDOW_S = 100.0

# metric, comparison, target. These match the plan's measured gates.
GATES = (
    ('id_switches_per_100s', '<=', 1.0),
    ('walker_recall', '>=', 0.90),
    ('position_rmse_m', '<=', 0.25),
    ('velocity_rmse_mps', '<=', 0.25),
    ('false_dynamic_rate', '<=', 0.01),
    ('time_to_dynamic_s', '<=', 1.5),
    ('ghosts', '<=', 0),
    ('rate_hz_p95', '>=', 2.0),
)


def read_frames(path):
    """Load a run log, oldest frame first."""
    frames = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                frames.append(json.loads(line))
    return sorted(frames, key=lambda frame: frame['stamp'])


def distance(track, state):
    return math.hypot(track['x'] - state[0], track['y'] - state[1])


def people(frame):
    return [track for track in frame['tracks'] if track['class'] == PERSON_CLASS]


def fit_time_offset(frames, walker, standing):
    """Find the shift between message stamps and the walker's script time."""
    fixed = standing.state_at(0.0)
    samples = [
        (frame['stamp'], track)
        for frame in frames
        for track in people(frame)
        if distance(track, fixed) > STATIC_RADIUS_M
    ]
    if not samples:
        return 0.0

    best_offset, best_cost = 0.0, math.inf
    for step in range(int(round(walker.period / OFFSET_STEP_S))):
        offset = step * OFFSET_STEP_S
        cost = sum(
            min(distance(track, walker.state_at(stamp + offset)) ** 2, 1.0)
            for stamp, track in samples
        )
        if cost < best_cost:
            best_offset, best_cost = offset, cost

    # The path loops, so report the shift as its smallest signed value
    if best_offset > walker.period / 2.0:
        best_offset -= walker.period
    return best_offset


def match_frame(frame, truths):
    """Pair each actor with its nearest person detection, closest pairs first."""
    candidates = sorted(
        (
            (distance(track, state), name, index)
            for name, state in truths.items()
            for index, track in enumerate(people(frame))
        ),
        key=lambda candidate: candidate[0],
    )
    matches, used = {}, set()
    for gap, name, index in candidates:
        if gap <= MATCH_RADIUS_M and name not in matches and index not in used:
            matches[name] = people(frame)[index]
            used.add(index)
    return matches


def rmse(errors):
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(math.ceil(fraction * len(ordered))) - 1, len(ordered) - 1)]


def count_id_switches(ids):
    known = [track_id for track_id in ids if track_id != -1]
    return sum(1 for before, after in zip(known, known[1:]) if before != after)


def time_to_dynamic(walker_matches):
    """Return the slowest first-seen to DYNAMIC delay across the walker's track IDs."""
    first_seen, first_dynamic = {}, {}
    for stamp, track in walker_matches:
        track_id = track['id']
        first_seen.setdefault(track_id, stamp)
        if track['state'] == 'DYNAMIC':
            first_dynamic.setdefault(track_id, stamp)
    delays = [stamp - first_seen[track_id] for track_id, stamp in first_dynamic.items()]
    return max(delays) if delays else None


def count_ghosts(frames, standing):
    """Count final world model entries beyond one that sits on the standing person."""
    if not frames:
        return 0
    entries = frames[-1]['world_model'].values()
    fixed = standing.state_at(0.0)
    on_truth = sum(
        1 for x, y, *_ in entries
        if math.hypot(x - fixed[0], y - fixed[1]) <= STATIC_RADIUS_M
    )
    return len(entries) - min(on_truth, 1)


def score(frames, walker, standing, time_offset=None):
    """Compute every metric for a run. A metric with no data is None."""
    if time_offset is None:
        time_offset = fit_time_offset(frames, walker, standing)

    walker_matches, standing_matches = [], []
    position_errors, velocity_errors, cruise_errors = [], [], []
    aligned = total_people = 0

    for frame in frames:
        stamp = frame['stamp']
        truth_time = stamp + time_offset
        walker_state = walker.state_at(truth_time)
        truths = {WALKER: walker_state, STANDING: standing.state_at(0.0)}
        matches = match_frame(frame, truths)

        for track in people(frame):
            total_people += 1
            if min(distance(track, state) for state in truths.values()) <= ALIGNMENT_RADIUS_M:
                aligned += 1

        for name, track in matches.items():
            position_errors.append(distance(track, truths[name]))

        if WALKER in matches:
            track = matches[WALKER]
            walker_matches.append((stamp, track))
            error = math.hypot(
                track['vx'] - walker_state[2], track['vy'] - walker_state[3]
            )
            velocity_errors.append(error)
            earlier = walker.state_at(truth_time - TURN_SETTLE_S)
            moving = math.hypot(walker_state[2], walker_state[3]) > 0.0
            if moving and earlier[2:] == walker_state[2:]:
                cruise_errors.append(error)

        if STANDING in matches:
            standing_matches.append(matches[STANDING])

    stamps = [frame['stamp'] for frame in frames]
    duration = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    intervals = [after - before for before, after in zip(stamps, stamps[1:])]
    slow_interval = percentile(intervals, 0.95)
    switches = count_id_switches([track['id'] for _, track in walker_matches])
    false_dynamic = sum(1 for track in standing_matches if track['state'] == 'DYNAMIC')

    return {
        'frames': len(frames),
        'duration_s': duration,
        'time_offset_s': time_offset,
        'aligned_detection_rate': aligned / total_people if total_people else None,
        'id_switches': switches,
        'id_switches_per_100s': (
            switches * ID_SWITCH_WINDOW_S / duration if duration > 0.0 else None
        ),
        'walker_recall': len(walker_matches) / len(frames) if frames else None,
        'standing_recall': len(standing_matches) / len(frames) if frames else None,
        'position_rmse_m': rmse(position_errors),
        'velocity_rmse_mps': rmse(velocity_errors),
        'velocity_rmse_cruise_mps': rmse(cruise_errors),
        'false_dynamic_rate': (
            false_dynamic / len(standing_matches) if standing_matches else None
        ),
        'time_to_dynamic_s': time_to_dynamic(walker_matches),
        'ghosts': count_ghosts(frames, standing),
        'rate_hz_p95': 1.0 / slow_interval if slow_interval else None,
        'latency_p95_s': percentile([frame['latency_s'] for frame in frames], 0.95),
    }


def failed_gates(metrics):
    """Return the names of gates that fail. A metric with no data fails."""
    failed = []
    for name, comparison, target in GATES:
        value = metrics[name]
        passed = value is not None and (
            value <= target if comparison == '<=' else value >= target
        )
        if not passed:
            failed.append(name)
    return failed


def format_value(value):
    if value is None:
        return 'no data'
    return f'{value:.3f}' if isinstance(value, float) else str(value)


def print_report(metrics):
    gates = {name: (comparison, target) for name, comparison, target in GATES}
    failed = failed_gates(metrics)
    for name, value in metrics.items():
        line = f'{name:28} {format_value(value):>10}'
        if name in gates:
            comparison, target = gates[name]
            verdict = 'FAIL' if name in failed else 'PASS'
            line += f'   {verdict}  (target {comparison} {target})'
        print(line)

    duration = metrics['duration_s']
    if 0.0 < duration < ID_SWITCH_WINDOW_S:
        print(
            f'\nNote: this run is {duration:.1f} s long, so a single ID switch counts as '
            f'{ID_SWITCH_WINDOW_S / duration:.1f} per 100 s.'
        )

    offset = metrics['time_offset_s']
    if abs(offset) > EXPECTED_OFFSET_S:
        print(
            f'\nNote: the walker fits best {offset:+.2f} s from sim time, more than the '
            f'expected {EXPECTED_OFFSET_S} s. Check aligned_detection_rate before trusting '
            'these scores.'
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run_log', help='File written through the run_log_path parameter')
    parser.add_argument('--world', default=str(DEFAULT_WORLD), help='SDF world file')
    parser.add_argument(
        '--time-offset', type=float, default=None,
        help='Seconds to add to stamps to get script time. Fitted from the run when omitted',
    )
    parser.add_argument('--gates', action='store_true', help='Exit 1 if any gate fails')
    parser.add_argument('--json', dest='json_path', help='Also write the metrics to this file')
    args = parser.parse_args(argv)

    frames = read_frames(args.run_log)
    if not frames:
        print(f'No frames in {args.run_log}', file=sys.stderr)
        return 2

    actors = load_actor_paths(args.world)
    metrics = score(frames, actors[WALKER], actors[STANDING], args.time_offset)
    print_report(metrics)

    if args.json_path:
        with open(args.json_path, 'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2)
            handle.write('\n')

    return 1 if args.gates and failed_gates(metrics) else 0


if __name__ == '__main__':
    sys.exit(main())
