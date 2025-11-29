#!/usr/bin/env python3
"""Capture error to file."""
import sys
import traceback
from pathlib import Path

error_file = Path(__file__).parent / 'error_capture.txt'

try:
    with open(error_file, 'w') as f:
        f.write("Starting app.py execution...\n")
        f.flush()
        
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        f.write(f"Project root: {project_root}\n")
        f.write(f"Python path: {sys.path[:3]}\n")
        f.flush()
        
        # Try importing app
        f.write("Attempting to import app module...\n")
        f.flush()
        
        import app
        f.write("App module imported successfully\n")
        f.flush()
        
        # Try running main
        f.write("Attempting to run main()...\n")
        f.flush()
        
        app.main()
        f.write("Main completed successfully\n")
        
except Exception as e:
    with open(error_file, 'a') as f:
        f.write(f"\nERROR OCCURRED:\n")
        f.write(f"{type(e).__name__}: {e}\n")
        f.write("\nFull traceback:\n")
        traceback.print_exc(file=f)
    raise

