#!/usr/bin/env python3
"""Diagnostic script to find errors."""
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"Python version: {sys.version}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")
print()

# Test basic imports
print("Testing basic imports...")
try:
    import numpy
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import pandas
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import yaml
    print("✓ yaml")
except Exception as e:
    print(f"✗ yaml: {e}")

try:
    import tqdm
    print("✓ tqdm")
except Exception as e:
    print(f"✗ tqdm: {e}")

print()
print("Testing project imports...")

try:
    from src.utils import setup_logging
    print("✓ src.utils")
except Exception as e:
    print(f"✗ src.utils: {e}")
    traceback.print_exc()

try:
    from src.data_generator import DataGenerator
    print("✓ src.data_generator")
except Exception as e:
    print(f"✗ src.data_generator: {e}")
    traceback.print_exc()

try:
    from src.experiment_runner import ExperimentRunner
    print("✓ src.experiment_runner")
except Exception as e:
    print(f"✗ src.experiment_runner: {e}")
    traceback.print_exc()

print()
print("Testing app import...")
try:
    import app
    print("✓ app module imported successfully")
except Exception as e:
    print(f"✗ app import failed: {e}")
    traceback.print_exc()

