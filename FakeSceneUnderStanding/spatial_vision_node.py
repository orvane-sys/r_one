#!/usr/bin/env python3
"""
Backwards-compatibility stub.
Delegates directly to r_one.spatial_vision.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from r_one.spatial_vision import SpatialVisionNode, main

if __name__ == '__main__':
    main()
