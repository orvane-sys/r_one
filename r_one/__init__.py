"""r_one: ROS 2 Spatial Perception and 3D Scene Understanding package."""

__version__ = '0.1.0'


# Lazy module access or explicit subpackage imports
def __getattr__(name):
    if name == 'SpatialVisionNode':
        from r_one.spatial_vision import SpatialVisionNode
        return SpatialVisionNode
    elif name == 'SituationalAwarenessNode':
        from r_one.situational_awareness import SituationalAwarenessNode
        return SituationalAwarenessNode
    elif name == 'SensorFusionNode':
        from r_one.sensor_fusion import SensorFusionNode
        return SensorFusionNode
    elif name == 'MockAINode':
        from r_one.mock_ai import MockAINode
        return MockAINode
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
