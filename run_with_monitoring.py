#!/usr/bin/env python3
"""Run app.py with enhanced error monitoring and logging."""
import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Redirect stdout/stderr to capture all output
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Create log file
log_file = project_root / 'run_log.txt'
with open(log_file, 'w', encoding='utf-8') as f:
    # Tee output to both console and file
    sys.stdout = TeeOutput(sys.stdout, f)
    sys.stderr = TeeOutput(sys.stderr, f)
    
    print("=" * 70)
    print("Research Experiment Engine - Starting with Monitoring")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python: {sys.executable} {sys.version}")
    print()
    
    # Quick Ollama status (non-blocking)
    print("Checking Ollama status (non-blocking)...")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=1)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"✓ Ollama running with models: {models}")
        else:
            print(f"⚠ Ollama status: {response.status_code}")
    except:
        print("⚠ Ollama check skipped (will be handled by app)")
    
    print()
    print("=" * 70)
    print("Importing and running app.py...")
    print("=" * 70)
    print()
    
    try:
        # Import and run the app
        import app
        
        # Check if app has main function
        if hasattr(app, 'main'):
            print("Calling app.main()...")
            app.main()
        else:
            print("App module imported but no main() function found")
            print("Trying to execute app.py directly...")
            exec(open('app.py').read())
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n" + "=" * 70)
        print("Execution completed. Check run_log.txt for full output.")
        print("=" * 70)

