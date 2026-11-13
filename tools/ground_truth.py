"""Work out where the demo's actors are at any simulation time, from the world file."""

from dataclasses import dataclass
from typing import Dict, Tuple
import xml.etree.ElementTree as ElementTree

# spatial_scene.launch.py spawns the robot here facing +x, so odom = world - spawn.
ROBOT_SPAWN_XY = (-2.0, -0.5)


@dataclass(frozen=True)
class ActorPath:
    """One actor's looping trajectory, with positions in the odom frame."""

    name: str
    times: Tuple[float, ...]
    xs: Tuple[float, ...]
    ys: Tuple[float, ...]
    delay: float = 0.0

    @property
    def period(self) -> float:
        return self.times[-1]

    def state_at(self, sim_time: float) -> Tuple[float, float, float, float]:
        """Return (x, y, vx, vy) at a simulation time."""
        phase = max(sim_time - self.delay, 0.0) % self.period
        for index in range(len(self.times) - 1):
            start, end = self.times[index], self.times[index + 1]
            if phase > end or end <= start:
                continue
            span = end - start
            fraction = (phase - start) / span
            dx = self.xs[index + 1] - self.xs[index]
            dy = self.ys[index + 1] - self.ys[index]
            return (
                self.xs[index] + fraction * dx,
                self.ys[index] + fraction * dy,
                dx / span,
                dy / span,
            )
        return (self.xs[-1], self.ys[-1], 0.0, 0.0)


def load_actor_paths(world_path, spawn_xy=ROBOT_SPAWN_XY) -> Dict[str, ActorPath]:
    """Read every actor's scripted trajectory from an SDF world file."""
    paths = {}
    for actor in ElementTree.parse(world_path).getroot().iter('actor'):
        name = actor.get('name')
        script = actor.find('script')
        trajectory = script.find('trajectory') if script is not None else None
        if trajectory is None:
            raise ValueError(f'Actor {name} has no scripted trajectory')

        waypoints = sorted(
            (
                float(waypoint.findtext('time')),
                [float(value) for value in waypoint.findtext('pose').split()],
            )
            for waypoint in trajectory.findall('waypoint')
        )
        if len(waypoints) < 2:
            raise ValueError(f'Actor {name} needs at least two waypoints')

        paths[name] = ActorPath(
            name=name,
            times=tuple(time for time, _ in waypoints),
            xs=tuple(pose[0] - spawn_xy[0] for _, pose in waypoints),
            ys=tuple(pose[1] - spawn_xy[1] for _, pose in waypoints),
            delay=float(script.findtext('delay_start', default='0')),
        )
    return paths
