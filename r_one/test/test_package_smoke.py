"""Smoke tests that import and exercise the real r_one code."""

import os
import unittest

from r_one import mock_ai, sensor_fusion, situational_awareness
from r_one.spatial_vision import resolve_model_path, Tracker, WorldModel


class TestModulesImport(unittest.TestCase):
    def test_nodes_expose_main(self):
        for module in (mock_ai, sensor_fusion, situational_awareness):
            self.assertTrue(callable(module.main), module.__name__)


class TestTrackerWiring(unittest.TestCase):
    def test_node_module_uses_the_tracker_and_world_model(self):
        # test_tracking.py covers both in depth; this only checks the node can reach them.
        self.assertEqual(Tracker().step(0.0, []), [])
        self.assertEqual(len(WorldModel()), 0)


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


if __name__ == '__main__':
    unittest.main()
