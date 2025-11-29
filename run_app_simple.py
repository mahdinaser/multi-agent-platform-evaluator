#!/usr/bin/env python3
"""Simple wrapper to run app.py with better error handling."""
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Starting Research Experiment Engine")
print("=" * 70)
print()

try:
    # Test imports
    print("Testing imports...")
    from src.utils import load_config, setup_logging
    print("✓ Imports OK")
    
    # Load config
    print("Loading config...")
    config = load_config('config/config.yaml')
    print("✓ Config loaded")
    
    # Run main
    print()
    print("Running app.py...")
    print()
    
    import app
    app.main()
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    sys.exit(1)
except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR OCCURRED:")
    print("=" * 70)
    print(f"{type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)

