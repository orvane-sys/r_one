"""Smoke tests that import and exercise the real r_one code and repository files."""

import importlib.util
import os
import unittest

import xacro
import yaml

from r_one import mock_ai, sensor_fusion, situational_awareness
from r_one.spatial_vision import KalmanTracker2D, resolve_model_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class TestModulesImport(unittest.TestCase):
    def test_nodes_expose_main(self):
        for module in (mock_ai, sensor_fusion, situational_awareness):
            self.assertTrue(callable(module.main), module.__name__)


class TestKalmanTracker2D(unittest.TestCase):
    def test_converges_to_constant_velocity(self):
        tracker = KalmanTracker2D(0.0, 0.0)
        dt = 0.1
        vx = 0.0
        for step in range(1, 41):
            tracker.predict(dt)
            _, _, vx, _ = tracker.update(1.0 * step * dt, 0.0)
        self.assertAlmostEqual(vx, 1.0, delta=0.1)

    def test_predict_counts_missed_frames_and_update_resets(self):
        tracker = KalmanTracker2D(0.0, 0.0)
        tracker.predict(0.1)
        tracker.predict(0.1)
        self.assertEqual(tracker.missed_frames, 2)
        tracker.update(0.0, 0.0)
        self.assertEqual(tracker.missed_frames, 0)


class TestResolveModelPath(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get('R_ONE_MODEL_DIR')

    def tearDown(self):
        if self._saved is None:
            os.environ.pop('R_ONE_MODEL_DIR', None)
        else:
            os.environ['R_ONE_MODEL_DIR'] = self._saved

    def test_unset_env_returns_name(self):
        os.environ.pop('R_ONE_MODEL_DIR', None)
        self.assertEqual(resolve_model_path('yolov8n.pt'), 'yolov8n.pt')

    def test_model_in_model_dir_is_preferred(self):
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir:
            weights = os.path.join(model_dir, 'yolov8n.pt')
            open(weights, 'wb').close()
            os.environ['R_ONE_MODEL_DIR'] = model_dir
            self.assertEqual(resolve_model_path('yolov8n.pt'), weights)

    def test_missing_file_and_absolute_paths_are_untouched(self):
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir:
            os.environ['R_ONE_MODEL_DIR'] = model_dir
            self.assertEqual(resolve_model_path('other.pt'), 'other.pt')
            self.assertEqual(resolve_model_path('/abs/custom.pt'), '/abs/custom.pt')


class TestRepositoryFiles(unittest.TestCase):
    def test_config_yaml_is_valid(self):
        config_dir = os.path.join(REPO_ROOT, 'config')
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        self.assertGreater(len(yaml_files), 0)
        for name in yaml_files:
            with open(os.path.join(config_dir, name)) as handle:
                try:
                    yaml.safe_load(handle)
                except yaml.YAMLError as exc:
                    self.fail(f'Invalid YAML in {name}: {exc}')

    def test_robot_description_renders(self):
        xacro_file = os.path.join(REPO_ROOT, 'description', 'robot.urdf.xacro')
        urdf = xacro.process_file(xacro_file).toxml()
        self.assertIn('<robot', urdf)
        self.assertIn('base_link', urdf)

    def test_launch_files_generate_a_description(self):
        launch_dir = os.path.join(REPO_ROOT, 'launch')
        launch_files = sorted(f for f in os.listdir(launch_dir) if f.endswith('.launch.py'))
        self.assertGreater(len(launch_files), 0)
        for name in launch_files:
            spec = importlib.util.spec_from_file_location(
                name.replace('.', '_'), os.path.join(launch_dir, name)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            description = module.generate_launch_description()
            self.assertTrue(len(description.entities) > 0, name)


if __name__ == '__main__':
    unittest.main()
