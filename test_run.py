#!/usr/bin/env python3
"""Test script to capture errors."""
import sys
import traceback

try:
    import app
    print("Import successful")
except Exception as e:
    print(f"ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

