"""
Track objects in world coordinates with constant-velocity Kalman filters.

Each frame, every track is predicted to the frame's time, matched to the new measurements
by Mahalanobis distance with the Hungarian method, then updated. A track is published once
it has been matched enough times, and ends when it has gone unmatched for too long.
"""

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

UNKNOWN = 'UNKNOWN'
STATIC = 'STATIC'
DYNAMIC = 'DYNAMIC'

# Pairs that must not match get this cost, so the assignment still has a finite solution.
NO_MATCH = 1e9

# The measurement is the position, the first two entries of the state [x, y, vx, vy].
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker tuning. The defaults were tuned on the sample run fixture and synthetic tests."""

    measurement_std_m: float = 0.2
    # White-noise acceleration: how quickly a track's velocity is allowed to change.
    accel_std_mps2: float = 2.0
    initial_velocity_std_mps: float = 1.0
    # The slower filter that only decides STATIC or DYNAMIC. Its small prior on speed keeps
    # a new track's first noisy steps from looking like motion.
    motion_accel_std_mps2: float = 0.1
    motion_initial_velocity_std_mps: float = 0.3
    # 99% of the chi-squared distribution with 2 degrees of freedom.
    gate_chi2: float = 9.21
    max_match_distance_m: float = 1.5
    max_coast_s: float = 2.5
    confirm_hits: int = 2
    speed_threshold_mps: float = 0.15
    # How many velocity standard deviations the speed must clear the threshold by to count
    # as DYNAMIC.
    motion_sigma_k: float = 1.0
    # Consecutive slow matches needed before a track counts as STATIC.
    static_updates: int = 5
    max_tracks: int = 64


@dataclass(frozen=True)
class Measurement:
    """One ranged detection in world coordinates."""

    x: float
    y: float
    z: float
    class_name: str
    score: float
    box: Tuple[float, ...] = ()


class ConstantVelocityFilter:
    """Kalman filter over [x, y, vx, vy] with white-noise acceleration."""

    def __init__(self, x: float, y: float, position_std: float, velocity_std: float):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([position_std ** 2, position_std ** 2,
                          velocity_std ** 2, velocity_std ** 2])

    @property
    def speed(self) -> float:
        return math.hypot(self.x[2], self.x[3])

    @property
    def velocity_std(self) -> float:
        return math.sqrt(max(self.P[2, 2], self.P[3, 3]))

    def predict(self, dt: float, accel_std: float):
        """Move the state dt seconds ahead. The uncertainty grows with dt."""
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2.0, 0.0], [0.0, dt * dt / 2.0], [dt, 0.0], [0.0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * accel_std ** 2

    def innovation(self, x: float, y: float, measurement_std: float):
        residual = np.array([x, y]) - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * measurement_std ** 2
        return residual, S

    def update(self, x: float, y: float, measurement_std: float):
        """Correct the state with a position (Joseph form, so P stays symmetric)."""
        residual, S = self.innovation(x, y, measurement_std)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        I_KH = np.eye(4) - K @ H
        R = np.eye(2) * measurement_std ** 2
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


class Track:
    """
    One object's filters and lifecycle.

    The main filter follows quick changes and gives the published position and velocity. A
    second, slower filter is used only to decide whether the object moves, because the main
    filter's velocity is too noisy to ever prove that an object stands still.
    """

    def __init__(self, track_id: int, measurement: Measurement, t: float,
                 config: TrackerConfig):
        self.id = track_id
        self.class_name = measurement.class_name
        self.measurement = measurement
        self.filter = ConstantVelocityFilter(
            measurement.x, measurement.y,
            config.measurement_std_m, config.initial_velocity_std_mps,
        )
        self.motion_filter = ConstantVelocityFilter(
            measurement.x, measurement.y,
            config.measurement_std_m, config.motion_initial_velocity_std_mps,
        )
        self.created = t
        self.last_update = t
        self.time = t
        self.hits = 1
        self.motion = UNKNOWN
        self.static_streak = 0

    @property
    def position(self) -> Tuple[float, float, float]:
        return float(self.filter.x[0]), float(self.filter.x[1]), float(self.measurement.z)

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.filter.x[2]), float(self.filter.x[3])

    @property
    def speed(self) -> float:
        return self.filter.speed

    def predict(self, t: float, config: TrackerConfig):
        """Move both filters to time t."""
        dt = t - self.time
        if dt <= 0.0:
            return
        self.filter.predict(dt, config.accel_std_mps2)
        self.motion_filter.predict(dt, config.motion_accel_std_mps2)
        self.time = t

    def mahalanobis(self, measurement: Measurement, measurement_std: float) -> float:
        """Return the squared Mahalanobis distance from the prediction to a measurement."""
        residual, S = self.filter.innovation(measurement.x, measurement.y, measurement_std)
        return float(residual @ np.linalg.solve(S, residual))

    def update(self, measurement: Measurement, measurement_std: float):
        """Correct both filters with a matched measurement."""
        self.filter.update(measurement.x, measurement.y, measurement_std)
        self.motion_filter.update(measurement.x, measurement.y, measurement_std)
        self.measurement = measurement
        self.hits += 1


class Tracker:
    """Match measurements to tracks frame by frame and decide which tracks move."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.tracks: List[Track] = []
        self._next_id = 1
        self._ended: List[int] = []
        self._matched_ids = set()

    def step(self, t: float, measurements: Sequence[Measurement]) -> List[Track]:
        """Process one frame and return the confirmed tracks matched in it."""
        config = self.config
        for track in self.tracks:
            track.predict(t, config)

        self._ended = [
            track.id for track in self.tracks if t - track.last_update > config.max_coast_s
        ]
        self.tracks = [
            track for track in self.tracks if t - track.last_update <= config.max_coast_s
        ]

        matched = []
        pairs, unmatched = self._associate(measurements)
        for track, measurement in pairs:
            track.update(measurement, config.measurement_std_m)
            track.last_update = t
            matched.append(track)
        for measurement in unmatched:
            track = Track(self._next_id, measurement, t, config)
            self._next_id += 1
            self.tracks.append(track)
            matched.append(track)

        for track in matched:
            self._decide_motion(track)
        self._enforce_capacity()

        live_ids = {track.id for track in self.tracks}
        self._matched_ids = {track.id for track in matched if track.id in live_ids}
        return [
            track for track in matched
            if track.id in live_ids and track.hits >= config.confirm_hits
        ]

    def coasting_dynamic(self) -> List[Track]:
        """Return confirmed moving tracks that were not matched in the last frame."""
        return [
            track for track in self.tracks
            if track.id not in self._matched_ids
            and track.hits >= self.config.confirm_hits
            and track.motion == DYNAMIC
        ]

    def ended_ids(self) -> List[int]:
        """Return the IDs of the tracks that ended in the last frame."""
        return list(self._ended)

    def _associate(self, measurements: Sequence[Measurement]):
        if not self.tracks or not measurements:
            return [], list(measurements)

        config = self.config
        cost = np.full((len(self.tracks), len(measurements)), NO_MATCH)
        for row, track in enumerate(self.tracks):
            for col, measurement in enumerate(measurements):
                if measurement.class_name != track.class_name:
                    continue
                x, y, _ = track.position
                gap = math.hypot(measurement.x - x, measurement.y - y)
                if gap > config.max_match_distance_m:
                    continue
                distance = track.mahalanobis(measurement, config.measurement_std_m)
                if distance <= config.gate_chi2:
                    cost[row, col] = distance

        pairs, used = [], set()
        for row, col in zip(*linear_sum_assignment(cost)):
            if cost[row, col] < NO_MATCH:
                pairs.append((self.tracks[row], measurements[col]))
                used.add(col)
        unmatched = [
            measurement for col, measurement in enumerate(measurements) if col not in used
        ]
        return pairs, unmatched

    def _decide_motion(self, track: Track):
        """Call a track DYNAMIC when it clearly moves, STATIC after a run of slow matches."""
        config = self.config
        speed = track.motion_filter.speed
        margin = config.motion_sigma_k * track.motion_filter.velocity_std
        if speed - margin > config.speed_threshold_mps:
            track.motion = DYNAMIC
            track.static_streak = 0
        elif speed < config.speed_threshold_mps:
            track.static_streak += 1
            if track.static_streak >= config.static_updates:
                track.motion = STATIC
        else:
            track.static_streak = 0

    def _enforce_capacity(self):
        """Drop the stalest tracks, then the least seen, when there are too many."""
        excess = len(self.tracks) - self.config.max_tracks
        if excess <= 0:
            return
        ordered = sorted(self.tracks, key=lambda track: (track.last_update, track.hits))
        dropped = {track.id for track in ordered[:excess]}
        self._ended.extend(sorted(dropped))
        self.tracks = [track for track in self.tracks if track.id not in dropped]
