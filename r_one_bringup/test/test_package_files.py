"""Check that the configuration, robot description and launch files load."""

import importlib.util
import os
import unittest

import xacro
import yaml

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class TestRepositoryFiles(unittest.TestCase):
    def test_config_yaml_is_valid(self):
        config_dir = os.path.join(PACKAGE_ROOT, 'config')
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        self.assertGreater(len(yaml_files), 0)
        for name in yaml_files:
            with open(os.path.join(config_dir, name)) as handle:
                try:
                    yaml.safe_load(handle)
                except yaml.YAMLError as exc:
                    self.fail(f'Invalid YAML in {name}: {exc}')

    def test_robot_description_renders(self):
        xacro_file = os.path.join(PACKAGE_ROOT, 'description', 'robot.urdf.xacro')
        urdf = xacro.process_file(xacro_file).toxml()
        self.assertIn('<robot', urdf)
        self.assertIn('base_link', urdf)

    def test_launch_files_generate_a_description(self):
        launch_dir = os.path.join(PACKAGE_ROOT, 'launch')
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
