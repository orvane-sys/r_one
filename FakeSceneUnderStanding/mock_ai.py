#!/usr/bin/env python3
"""
Backwards-compatibility stub.
Delegates directly to r_one.mock_ai.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from r_one.mock_ai import MockAINode, main

if __name__ == '__main__':
    main()
