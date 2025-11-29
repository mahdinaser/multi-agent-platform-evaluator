#!/usr/bin/env python3
"""Simple test to verify Python execution."""
print("=" * 80)
print("Simple Test Script")
print("=" * 80)
print(f"Python version: {__import__('sys').version}")
print("Testing imports...")

try:
    import numpy
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy failed: {e}")

try:
    import pandas
    print("✓ pandas imported")
except ImportError as e:
    print(f"✗ pandas failed: {e}")

try:
    import yaml
    print("✓ yaml imported")
except ImportError as e:
    print(f"✗ yaml failed: {e}")

try:
    from pathlib import Path
    import sys
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    from src.utils import setup_logging
    print("✓ src.utils imported")
except Exception as e:
    print(f"✗ src.utils failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("Test complete")

