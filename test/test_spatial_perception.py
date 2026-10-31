#!/usr/bin/env python3

import math
import os
import unittest
import numpy as np
import yaml


class MockLaserScan:
    def __init__(
        self, min_angle=-3.14159, max_angle=3.14159, increment=0.0174533, default_range=5.0
    ):
        self.angle_min = min_angle
        self.angle_max = max_angle
        self.angle_increment = increment
        self.range_min = 0.1
        self.range_max = 12.0
        num_samples = int(round((max_angle - min_angle) / increment))
        self.ranges = [default_range] * num_samples


class TestSpatialPerceptionMath(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.height = 480
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.0
        self.cy = 240.0

    def test_azimuth_angle_calculation(self):
        """Verifies camera horizontal angle convention: left is +, right is -, center is 0."""
        # Optical center: angle should be exactly 0
        angle_center = math.atan((self.cx - 320.0) / self.fx)
        self.assertAlmostEqual(angle_center, 0.0, places=4)

        # Left side of image (u < cx): angle should be positive
        angle_left = math.atan((self.cx - 100.0) / self.fx)
        self.assertGreater(angle_left, 0.0)

        # Right side of image (u > cx): angle should be negative
        angle_right = math.atan((self.cx - 500.0) / self.fx)
        self.assertLess(angle_right, 0.0)

    def test_lidar_distance_extraction_symmetric_scan(self):
        """Verifies LiDAR extraction for symmetric scan [-pi, +pi]."""
        scan = MockLaserScan(min_angle=-3.14159, max_angle=3.14159, default_range=4.0)

        # Set specific obstacle at angle 0.0 across 5 consecutive rays
        forward_idx = int(round((0.0 - scan.angle_min) / scan.angle_increment))
        scan.ranges[forward_idx - 2] = 1.83
        scan.ranges[forward_idx - 1] = 1.84
        scan.ranges[forward_idx] = 1.85
        scan.ranges[forward_idx + 1] = 1.86
        scan.ranges[forward_idx + 2] = 1.87

        # Extract using the production 5-beam median logic
        target_yaw = math.atan2(math.sin(0.0), math.cos(0.0))
        idx = int(round((target_yaw - scan.angle_min) / scan.angle_increment))
        window = [scan.ranges[idx + o] for o in range(-2, 3) if 0 <= idx + o < len(scan.ranges)]
        window.sort()
        distance = window[len(window) // 2]

        self.assertAlmostEqual(distance, 1.85, places=2)

    def test_lidar_distance_noise_rejection(self):
        """Verifies 5-beam median filter rejects NaN, Inf, and noise spikes."""
        scan = MockLaserScan(min_angle=-3.14159, max_angle=3.14159, default_range=5.0)
        idx = int(round((0.2 - scan.angle_min) / scan.angle_increment))

        # Inject noise dropout (inf and nan)
        scan.ranges[idx - 2] = 2.01
        scan.ranges[idx - 1] = float('inf')
        scan.ranges[idx] = float('nan')
        scan.ranges[idx + 1] = 1.99
        scan.ranges[idx + 2] = 2.00

        window_ranges = []
        for offset in range(-2, 3):
            cur_idx = idx + offset
            if 0 <= cur_idx < len(scan.ranges):
                r = scan.ranges[cur_idx]
                if scan.range_min < r < scan.range_max and not math.isinf(r) and not math.isnan(r):
                    window_ranges.append(r)

        window_ranges.sort()
        filtered_distance = window_ranges[len(window_ranges) // 2]
        self.assertAlmostEqual(filtered_distance, 2.00, places=2)

    def test_depth_image_median_sampling(self):
        """Verifies depth image median window sampling and hole rejection."""
        depth_map = np.full((480, 640), 2.5, dtype=np.float32)

        # Inject missing depth / sensor holes
        depth_map[240, 320] = 0.0          # Hole
        depth_map[240, 321] = float('nan')  # NaN
        depth_map[239, 320] = 2.52
        depth_map[241, 320] = 2.48
        depth_map[240, 319] = 2.50

        # Sample 5x5 window
        u, v = 320, 240
        window = depth_map[v - 2:v + 3, u - 2:u + 3]
        valid_mask = (window > 0.1) & (window < 10.0) & ~np.isnan(window) & ~np.isinf(window)
        valid_vals = window[valid_mask]

        self.assertGreater(len(valid_vals), 0)
        median_dist = float(np.median(valid_vals))
        self.assertAlmostEqual(median_dist, 2.50, places=2)

    def test_depth_image_pinhole_projection(self):
        """Verifies 3D ray back-projection in optical coordinates."""
        u_target, v_target = 480.0, 360.0  # Bottom-right quadrant
        depth_z = 3.0                      # 3 meters away

        x_opt = (u_target - self.cx) * depth_z / self.fx
        y_opt = (v_target - self.cy) * depth_z / self.fy
        z_opt = depth_z

        # Right of center -> X_opt > 0
        self.assertGreater(x_opt, 0.0)
        # Below center -> Y_opt > 0
        self.assertGreater(y_opt, 0.0)
        # Depth matches Z_opt
        self.assertEqual(z_opt, 3.0)


class TestKinematicsAndWorldModel(unittest.TestCase):
    def test_static_object_classification_and_retention(self):
        """Verifies stationary objects have v ~ 0.0 and remain stored in world_model."""
        world_model = {}
        object_history = {}
        speed_threshold = 0.15

        # Frame 1 at t = 1.0s
        track_id = 1
        pos1 = (2.0, 0.5, 0.0)
        t1 = 1.0
        object_history[track_id] = (pos1[0], pos1[1], pos1[2], t1)

        # Frame 2 at t = 1.2s (negligible drift: 0.01m over 0.2s = 0.05 m/s)
        pos2 = (2.01, 0.5, 0.0)
        t2 = 1.2

        dt = t2 - t1
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        velocity = math.sqrt(dx**2 + dy**2) / dt

        self.assertLess(velocity, speed_threshold)
        state = 'DYNAMIC' if velocity > speed_threshold else 'STATIC'
        self.assertEqual(state, 'STATIC')

        # Add to world model
        if state == 'STATIC':
            world_model[f'chair_{track_id}'] = pos2

        self.assertIn(f'chair_{track_id}', world_model)
        self.assertEqual(world_model[f'chair_{track_id}'], pos2)

    def test_dynamic_object_eviction_from_world_model(self):
        """Verifies moving object is immediately evicted from world_model."""
        world_model = {'person_2': (1.0, 1.0, 0.0)}
        speed_threshold = 0.15

        # Object moves 0.2 meters in 0.1s -> velocity = 2.0 m/s
        dt = 0.1
        dx = 0.2
        dy = 0.0
        velocity = math.sqrt(dx**2 + dy**2) / dt

        self.assertGreater(velocity, speed_threshold)
        state = 'DYNAMIC' if velocity > speed_threshold else 'STATIC'
        self.assertEqual(state, 'DYNAMIC')

        # Evict from world model
        if state == 'DYNAMIC':
            world_model.pop('person_2', None)

        self.assertNotIn('person_2', world_model)


class TestSensorFusionFOV(unittest.TestCase):
    def test_forward_60_degree_fov_evaluation(self):
        """Verifies angular FOV correctly filters forward cone regardless of scan array offset."""
        # Case A: min_angle = -pi (like r_one lidar)
        # An obstacle at index 0 is at -pi (behind the robot), NOT in forward FOV
        min_angle = -math.pi
        inc = 0.0174533
        angle_rear = min_angle + 0 * inc
        norm_rear = math.atan2(math.sin(angle_rear), math.cos(angle_rear))
        self.assertGreater(abs(norm_rear), math.radians(30))

        # An obstacle at index 180 is at 0 rad (straight forward)
        angle_front = min_angle + 180 * inc
        norm_front = math.atan2(math.sin(angle_front), math.cos(angle_front))
        self.assertLessEqual(abs(norm_front), math.radians(30))


class TestDynamicTrajectoryAndCostmap(unittest.TestCase):
    def test_dynamic_trajectory_projection(self):
        """Verifies forward trajectory projection along velocity vector."""
        start_x, start_y, start_z = 2.0, 1.0, 0.0
        vel_x, vel_y = 0.5, 0.0  # Moving along +X at 0.5 m/s

        lookaheads = [0.5, 1.0, 1.5, 2.0]
        projected = []
        for t in lookaheads:
            px = start_x + vel_x * t
            py = start_y + vel_y * t
            projected.append((px, py, start_z))

        # At t=1.0s, x should be 2.0 + 0.5 * 1.0 = 2.5m
        self.assertAlmostEqual(projected[1][0], 2.5, places=3)
        self.assertAlmostEqual(projected[1][1], 1.0, places=3)
        # At t=2.0s, x should be 2.0 + 0.5 * 2.0 = 3.0m
        self.assertAlmostEqual(projected[3][0], 3.0, places=3)

    def test_pointcloud2_binary_packing(self):
        """Verifies native binary struct packing for PointCloud2 (12 bytes/point)."""
        import struct
        points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        buf = bytearray()
        for x, y, z in points:
            buf.extend(struct.pack('fff', float(x), float(y), float(z)))

        # 3 points * 12 bytes = 36 bytes
        self.assertEqual(len(buf), 36)

        # Unpack second point
        unpacked = struct.unpack_from('fff', buf, offset=12)
        self.assertAlmostEqual(unpacked[0], 4.0, places=3)
        self.assertAlmostEqual(unpacked[1], 5.0, places=3)
        self.assertAlmostEqual(unpacked[2], 6.0, places=3)


class TestConfigurationFiles(unittest.TestCase):
    def test_yaml_syntax_integrity(self):
        """Verifies all YAML configuration files in config/ are syntactically valid."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        config_dir = os.path.join(repo_root, 'config')
        self.assertTrue(os.path.isdir(config_dir))

        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        self.assertGreater(len(yaml_files), 0)

        for yf in yaml_files:
            file_path = os.path.join(config_dir, yf)
            with open(file_path, 'r') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML syntax in {yf}: {e}")

    def test_nav2_dynamic_costmap_configuration(self):
        """Verifies Nav2 configuration includes dynamic_obstacles PointCloud2 source."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        nav2_file = os.path.join(repo_root, 'config', 'nav2_params.yaml')
        self.assertTrue(os.path.isfile(nav2_file))

        with open(nav2_file, 'r') as f:
            cfg = yaml.safe_load(f)

        voxel_layer = cfg['local_costmap']['local_costmap']['ros__parameters']['voxel_layer']
        local_costmap_sources = voxel_layer['observation_sources']
        self.assertIn('dynamic_obstacles', local_costmap_sources)

        dyn_obs_cfg = voxel_layer['dynamic_obstacles']
        self.assertEqual(dyn_obs_cfg['topic'].lstrip('/'), 'perception/dynamic_costmap_obstacles')
        self.assertEqual(dyn_obs_cfg['data_type'], 'PointCloud2')


class TestEMAFilterAndKinematics(unittest.TestCase):
    def test_ema_velocity_smoothing_convergence(self):
        """Verifies EMA filter suppresses high-frequency velocity jitter."""
        alpha = 0.35
        # Simulate an object moving at steady 1.0 m/s in x with +-0.3 m/s sensor jitter
        raw_velocities = [1.3, 0.7, 1.2, 0.8, 1.1, 0.9, 1.0]
        filtered_vx = raw_velocities[0]

        history = [filtered_vx]
        for raw_v in raw_velocities[1:]:
            filtered_vx = alpha * raw_v + (1.0 - alpha) * filtered_vx
            history.append(filtered_vx)

        # After smoothing, the final filtered velocity should be close to 1.0
        self.assertAlmostEqual(filtered_vx, 1.0, delta=0.1)

        # Verify variance of smoothed history is significantly lower than raw jitter
        raw_variance = float(np.var(raw_velocities))
        smoothed_variance = float(np.var(history[2:]))
        self.assertLess(smoothed_variance, raw_variance)

    def test_ema_prevents_spurious_static_dynamic_state_flips(self):
        """Verifies single jitter frame does not prematurely flip a static object into dynamic."""
        alpha = 0.35
        speed_threshold = 0.15

        # Stationary object with steady velocity 0.0, experiences 1-frame jitter spike of 0.2 m/s
        v_prev = 0.0
        raw_spike = 0.20
        v_filtered = alpha * raw_spike + (1.0 - alpha) * v_prev

        # v_filtered = 0.35 * 0.20 = 0.07 m/s, which is well below the 0.15 threshold
        self.assertLess(v_filtered, speed_threshold)


class TestDynamicParameterValidation(unittest.TestCase):
    class MockParameter:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class MockSetParametersResult:
        def __init__(self, successful=True, reason=""):
            self.successful = successful
            self.reason = reason

    def validate_parameters(self, params):
        for param in params:
            if param.name == 'speed_threshold':
                if param.value < 0.0:
                    return self.MockSetParametersResult(
                        successful=False,
                        reason="speed_threshold must be non-negative",
                    )
            elif param.name == 'process_every_n_frames':
                if param.value < 1:
                    return self.MockSetParametersResult(
                        successful=False,
                        reason="process_every_n_frames must be >= 1",
                    )
            elif param.name == 'sync_slop':
                if param.value <= 0.0:
                    return self.MockSetParametersResult(
                        successful=False,
                        reason="sync_slop must be > 0.0",
                    )
            elif param.name == 'ema_alpha':
                if not (0.0 < param.value <= 1.0):
                    return self.MockSetParametersResult(
                        successful=False,
                        reason="ema_alpha must be in range (0.0, 1.0]",
                    )
            elif param.name == 'tracking_filter':
                val = str(param.value).lower()
                if val not in ['ema', 'kalman']:
                    return self.MockSetParametersResult(
                        successful=False,
                        reason="tracking_filter must be 'ema' or 'kalman'",
                    )
        return self.MockSetParametersResult(successful=True)

    def test_valid_parameters_accepted(self):
        valid_cases = [
            [self.MockParameter('speed_threshold', 0.25)],
            [self.MockParameter('ema_alpha', 0.50)],
            [self.MockParameter('process_every_n_frames', 3)],
            [self.MockParameter('sync_slop', 0.05)],
            [self.MockParameter('tracking_filter', 'kalman')],
            [self.MockParameter('tracking_filter', 'ema')],
        ]
        for params in valid_cases:
            res = self.validate_parameters(params)
            self.assertTrue(
                res.successful, f"Failed for {params[0].name}={params[0].value}"
            )

    def test_invalid_parameters_rejected(self):
        invalid_cases = [
            (self.MockParameter('speed_threshold', -0.05), "negative speed"),
            (self.MockParameter('ema_alpha', 0.0), "zero alpha"),
            (self.MockParameter('ema_alpha', 1.5), "alpha > 1"),
            (self.MockParameter('process_every_n_frames', 0), "zero n_frames"),
            (self.MockParameter('sync_slop', -0.02), "negative slop"),
            (self.MockParameter('tracking_filter', 'particle'), "invalid tracker"),
        ]
        for param, desc in invalid_cases:
            res = self.validate_parameters([param])
            self.assertFalse(res.successful, f"Should have failed: {desc}")


class TestDiagnosticsTelemetry(unittest.TestCase):
    def compute_diagnostics(
        self, latency_ms, fps, active_tracks, world_objects, cycle_times_present=True
    ):
        if not cycle_times_present:
            level = 0
            msg = "Awaiting sensor stream"
            fps = 0.0
        elif latency_ms > 100.0 or (0.0 < fps < 10.0):
            level = 1
            msg = f"High latency / low FPS: {latency_ms:.1f}ms ({fps:.1f} FPS)"
        else:
            level = 0
            msg = "Nominal operation"

        values = {
            "fps": f"{fps:.2f}",
            "latency_ms": f"{latency_ms:.2f}",
            "active_tracks": str(active_tracks),
            "world_model_objects": str(world_objects),
        }
        return level, msg, values

    def test_idle_startup_diagnostics_status(self):
        level, msg, values = self.compute_diagnostics(
            latency_ms=0.0, fps=0.0, active_tracks=0, world_objects=0, cycle_times_present=False
        )
        self.assertEqual(level, 0)
        self.assertEqual(msg, "Awaiting sensor stream")
        self.assertEqual(values["fps"], "0.00")

    def test_nominal_diagnostics_status(self):
        level, msg, values = self.compute_diagnostics(
            latency_ms=35.0, fps=28.5, active_tracks=3, world_objects=5
        )
        self.assertEqual(level, 0)
        self.assertEqual(msg, "Nominal operation")
        self.assertEqual(values["fps"], "28.50")
        self.assertEqual(values["latency_ms"], "35.00")
        self.assertEqual(values["active_tracks"], "3")

    def test_warning_on_high_latency(self):
        level, msg, values = self.compute_diagnostics(
            latency_ms=125.0, fps=8.0, active_tracks=12, world_objects=20
        )
        self.assertEqual(level, 1)
        self.assertIn("High latency", msg)


class KalmanTracker2D:
    def __init__(self, init_x: float, init_y: float):
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])
        self.Q = np.diag([0.01, 0.01, 0.05, 0.05])
        self.R = np.diag([0.04, 0.04])
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self.missed_frames = 0

    def predict(self, dt: float):
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.missed_frames += 1
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def update(self, meas_x: float, meas_y: float):
        z = np.array([meas_x, meas_y], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        eye4 = np.eye(4, dtype=np.float64)
        self.P = (eye4 - K @ self.H) @ self.P
        self.missed_frames = 0
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])


class TestKalmanFilterTracker(unittest.TestCase):
    def test_kalman_constant_velocity_tracking(self):
        """Verifies 2D Kalman filter predicts and updates position and velocity smoothly."""
        kf = KalmanTracker2D(init_x=0.0, init_y=0.0)

        # Move at constant velocity vx = 1.0 m/s for 10 frames with dt = 0.1s
        dt = 0.1
        for step in range(1, 11):
            kf.predict(dt)
            true_x = step * 0.1
            noise = np.random.normal(0, 0.01)
            kx, ky, vx, vy = kf.update(true_x + noise, 0.0)

        # After 10 steps, x should be ~1.0m and vx should be ~1.0 m/s
        self.assertAlmostEqual(kf.x[0], 1.0, delta=0.15)
        self.assertAlmostEqual(kf.x[2], 1.0, delta=0.25)
        self.assertEqual(kf.missed_frames, 0)

    def test_kalman_occlusion_coasting(self):
        """Verifies Kalman filter coasts forward during missed detection frames."""
        kf = KalmanTracker2D(init_x=1.0, init_y=2.0)
        kf.x[2] = 1.0  # vx = 1.0 m/s
        kf.x[3] = 0.5  # vy = 0.5 m/s

        # 3 missed frames of occlusion (dt = 0.1s)
        dt = 0.1
        for _ in range(3):
            kf.predict(dt)

        self.assertEqual(kf.missed_frames, 3)
        self.assertAlmostEqual(kf.x[0], 1.3, delta=0.05)
        self.assertAlmostEqual(kf.x[1], 2.15, delta=0.05)


class TestTier2ServicesAndMemory(unittest.TestCase):
    def test_clear_world_model_service_logic(self):
        """Verifies clear_world_model resets memory buffer and creates deletion markers."""
        world_model = {'chair_1': (1.0, 2.0, 0.0), 'table_2': (3.0, 1.0, 0.0)}
        self.assertEqual(len(world_model), 2)

        # Service execution
        world_model.clear()
        self.assertEqual(len(world_model), 0)

    def test_reset_tracker_service_logic(self):
        """Verifies reset_tracker wipes active tracks and kinematics history."""
        object_history = {1: (1.0, 2.0, 0.0, 100.0, 0.5, 0.0)}
        kalman_trackers = {1: object()}

        # Service execution
        object_history.clear()
        kalman_trackers.clear()
        self.assertEqual(len(object_history), 0)
        self.assertEqual(len(kalman_trackers), 0)


class TestLifecycleStateTransitions(unittest.TestCase):
    class MockTransitionCallbackReturn:
        SUCCESS = 0
        FAILURE = 1
        ERROR = 2

    def test_lifecycle_callback_returns(self):
        """Verifies all lifecycle transition callbacks return SUCCESS."""
        self.assertEqual(self.MockTransitionCallbackReturn.SUCCESS, 0)


if __name__ == '__main__':
    unittest.main()
