"""Tests for the run log and for the scoring tools in the repository's tools/ folder."""

import json
import os
import sys
import tempfile
import unittest

from r_one.run_log import RunLog

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))

from ground_truth import load_actor_paths  # noqa: E402, I100, I202
import score_run  # noqa: E402, I100

WORLD = os.path.join(REPO_ROOT, 'r_one_bringup', 'worlds', 'spatial_demo.world')
ACTORS = load_actor_paths(WORLD)
WALKER = ACTORS[score_run.WALKER]
STANDING = ACTORS[score_run.STANDING]


def track(track_id, state, label):
    return {
        'id': track_id,
        'class': 'person',
        'score': 0.9,
        'x': state[0],
        'y': state[1],
        'z': 0.0,
        'vx': state[2],
        'vy': state[3],
        'state': label,
        'box': [0.0, 0.0, 1.0, 1.0],
    }


def perfect_frames(duration=30.0, step=0.2, truth_shift=0.0):
    """Build a run log from a tracker that always reports the exact truth."""
    still = STANDING.state_at(0.0)
    frames = []
    for index in range(int(duration / step)):
        stamp = index * step
        frames.append({
            'stamp': stamp,
            'latency_s': 0.05,
            'tracks': [
                track(1, WALKER.state_at(stamp + truth_shift), 'DYNAMIC'),
                track(2, still, 'STATIC'),
            ],
            'world_model': {'person_2': [still[0], still[1], 0.0]},
        })
    return frames


def score(frames):
    return score_run.score(frames, WALKER, STANDING)


class TestGroundTruth(unittest.TestCase):
    def test_standing_person_matches_the_observed_odom_position(self):
        x, y, vx, vy = STANDING.state_at(12.0)
        self.assertAlmostEqual(x, 4.5)
        self.assertAlmostEqual(y, -1.9)
        self.assertEqual((vx, vy), (0.0, 0.0))

    def test_walker_moves_at_the_scripted_speed(self):
        x, y, vx, vy = WALKER.state_at(2.0)
        self.assertAlmostEqual(x, 4.5)
        self.assertAlmostEqual(y, 0.5)
        self.assertAlmostEqual(vy, 0.75)
        self.assertAlmostEqual(vx, 0.0)

    def test_walker_pauses_then_returns(self):
        self.assertAlmostEqual(WALKER.state_at(4.25)[3], 0.0)
        self.assertAlmostEqual(WALKER.state_at(6.5)[3], -0.75)

    def test_walker_path_loops(self):
        self.assertEqual(WALKER.period, 9.0)
        self.assertAlmostEqual(WALKER.state_at(11.0)[1], WALKER.state_at(2.0)[1])


class TestScoring(unittest.TestCase):
    def test_perfect_run_passes_every_gate(self):
        metrics = score(perfect_frames())
        self.assertEqual(score_run.failed_gates(metrics), [])
        self.assertAlmostEqual(metrics['position_rmse_m'], 0.0)
        self.assertAlmostEqual(metrics['velocity_rmse_mps'], 0.0)
        self.assertEqual(metrics['id_switches'], 0)

    def test_time_offset_is_recovered(self):
        metrics = score(perfect_frames(truth_shift=1.2))
        self.assertAlmostEqual(metrics['time_offset_s'], 1.2, delta=score_run.OFFSET_STEP_S)
        self.assertLess(metrics['position_rmse_m'], 0.05)

    def test_time_offset_uses_the_smallest_signed_shift(self):
        metrics = score(perfect_frames(truth_shift=8.0))
        self.assertAlmostEqual(metrics['time_offset_s'], -1.0, delta=score_run.OFFSET_STEP_S)

    def test_id_switches_are_counted(self):
        frames = perfect_frames()
        for frame in frames[10:20]:
            frame['tracks'][0]['id'] = 7
        for frame in frames[20:]:
            frame['tracks'][0]['id'] = 8
        metrics = score(frames)
        self.assertEqual(metrics['id_switches'], 2)
        self.assertIn('id_switches_per_100s', score_run.failed_gates(metrics))

    def test_untracked_detections_do_not_count_as_switches(self):
        self.assertEqual(score_run.count_id_switches([1, -1, 1, -1, 1]), 0)

    def test_missed_walker_frames_lower_recall(self):
        frames = perfect_frames()
        for frame in frames[::2]:
            frame['tracks'].pop(0)
        metrics = score(frames)
        self.assertAlmostEqual(metrics['walker_recall'], 0.5, delta=0.01)
        self.assertIn('walker_recall', score_run.failed_gates(metrics))

    def test_position_error_is_measured(self):
        frames = perfect_frames()
        for frame in frames:
            for item in frame['tracks']:
                item['x'] += 0.3
        self.assertAlmostEqual(score(frames)['position_rmse_m'], 0.3, delta=0.01)

    def test_static_person_reported_dynamic_is_counted(self):
        frames = perfect_frames()
        for frame in frames[:15]:
            frame['tracks'][1]['state'] = 'DYNAMIC'
        metrics = score(frames)
        self.assertAlmostEqual(metrics['false_dynamic_rate'], 0.1)
        self.assertIn('false_dynamic_rate', score_run.failed_gates(metrics))

    def test_delay_before_dynamic_is_measured(self):
        frames = perfect_frames()
        for frame in frames[:10]:
            frame['tracks'][0]['state'] = 'STATIC'
        self.assertAlmostEqual(score(frames)['time_to_dynamic_s'], 2.0)

    def test_world_model_entry_on_the_walker_path_is_a_ghost(self):
        frames = perfect_frames()
        frames[-1]['world_model']['person_9'] = [4.5, 0.5, 0.0]
        metrics = score(frames)
        self.assertEqual(metrics['ghosts'], 1)
        self.assertIn('ghosts', score_run.failed_gates(metrics))

    def test_duplicate_entries_on_the_standing_person_are_ghosts(self):
        frames = perfect_frames()
        frames[-1]['world_model']['person_3'] = [4.5, -1.9, 0.0]
        self.assertEqual(score(frames)['ghosts'], 1)

    def test_empty_detection_run_has_no_metrics_and_fails(self):
        frames = perfect_frames()
        for frame in frames:
            frame['tracks'] = []
            frame['world_model'] = {}
        metrics = score(frames)
        self.assertIsNone(metrics['position_rmse_m'])
        self.assertIn('position_rmse_m', score_run.failed_gates(metrics))


class TestSampleRun(unittest.TestCase):
    """The run log recorded from the sample bag by the 2a baseline code."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_run_baseline.jsonl')
        cls.metrics = score(score_run.read_frames(path))

    def test_walker_script_lines_up_with_sim_time(self):
        self.assertLessEqual(abs(self.metrics['time_offset_s']), score_run.OFFSET_STEP_S)

    def test_every_detection_sits_on_an_actor(self):
        self.assertGreaterEqual(self.metrics['aligned_detection_rate'], 0.95)

    def test_both_actors_are_seen(self):
        self.assertGreater(self.metrics['walker_recall'], 0.8)
        self.assertGreater(self.metrics['standing_recall'], 0.95)


class TestRunLog(unittest.TestCase):
    def test_disabled_log_ignores_writes(self):
        log = RunLog()
        self.assertFalse(log.enabled)
        log.write_frame(1.0, 0.1, [], {})
        log.close()

    def test_log_round_trips_through_the_scorer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'run.jsonl')
            log = RunLog(path)
            self.assertTrue(log.enabled)
            still = STANDING.state_at(0.0)
            log.write_frame(
                30.0, 0.05, [track(2, still, 'STATIC')], {'person_2': (still[0], still[1], 0.0)}
            )
            log.close()
            log.close()

            frames = score_run.read_frames(path)
        self.assertEqual(frames[0]['stamp'], 30.0)
        self.assertEqual(frames[0]['tracks'][0]['id'], 2)
        self.assertEqual(frames[0]['world_model']['person_2'], [still[0], still[1], 0.0])

    def test_lines_are_flushed_before_close(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'run.jsonl')
            log = RunLog(path)
            log.write_frame(1.0, 0.1, [], {})
            with open(path, encoding='utf-8') as handle:
                self.assertEqual(json.loads(handle.readline())['stamp'], 1.0)
            log.close()


class TestCommandLine(unittest.TestCase):
    def write_run(self, folder, frames):
        path = os.path.join(folder, 'run.jsonl')
        with open(path, 'w', encoding='utf-8') as handle:
            for frame in frames:
                handle.write(json.dumps(frame) + '\n')
        return path

    def test_gates_set_the_exit_code(self):
        with tempfile.TemporaryDirectory() as folder:
            good = self.write_run(folder, perfect_frames())
            self.assertEqual(score_run.main([good, '--gates', '--world', WORLD]), 0)

            frames = perfect_frames()
            for frame in frames:
                frame['tracks'][1]['state'] = 'DYNAMIC'
            bad = self.write_run(folder, frames)
            self.assertEqual(score_run.main([bad, '--gates', '--world', WORLD]), 1)
            self.assertEqual(score_run.main([bad, '--world', WORLD]), 0)

    def test_metrics_can_be_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.write_run(folder, perfect_frames())
            output = os.path.join(folder, 'metrics.json')
            score_run.main([run, '--world', WORLD, '--json', output])
            with open(output, encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['id_switches'], 0)


if __name__ == '__main__':
    unittest.main()
