"""Rules for the spatial_vision parameters that can be checked on their own."""

from typing import Optional

# name: (predicate, reason). NaN fails every predicate, so it is rejected too.
PARAMETER_RULES = {
    'speed_threshold': (
        lambda value: value >= 0.0,
        "speed_threshold must be non-negative",
    ),
    'process_every_n_frames': (
        lambda value: value >= 1,
        "process_every_n_frames must be >= 1",
    ),
    'img_size': (
        lambda value: value >= 1,
        "img_size must be >= 1",
    ),
    'sync_slop': (
        lambda value: value > 0.0,
        "sync_slop must be > 0.0",
    ),
    'depth_source': (
        lambda value: value in ('lidar', 'depth_image'),
        "depth_source must be 'lidar' or 'depth_image'",
    ),
}


def validate_parameter(name: str, value) -> Optional[str]:
    """Return why a parameter value is invalid, or None when it is fine."""
    rule = PARAMETER_RULES.get(name)
    if rule is None or rule[0](value):
        return None
    return rule[1]
