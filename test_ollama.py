#!/usr/bin/env python3
"""Test script to check Ollama connection."""
import sys
import requests
import subprocess

print("=" * 60)
print("Ollama Server Test")
print("=" * 60)
print()

# Check if ollama command exists
try:
    result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"✓ Ollama installed: {result.stdout.strip()}")
    else:
        print("✗ Ollama command found but returned error")
        print(f"  Error: {result.stderr}")
        sys.exit(1)
except FileNotFoundError:
    print("✗ Ollama is not installed or not in PATH")
    print()
    print("Please install Ollama:")
    print("  1. Download from: https://ollama.ai/")
    print("  2. Install the application")
    print("  3. Add to PATH or restart terminal")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error checking Ollama: {e}")
    sys.exit(1)

# Check if server is running
print()
print("Checking if Ollama server is running...")
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=2)
    if response.status_code == 200:
        print("✓ Ollama server is running!")
        print()
        print("Available models:")
        models = response.json().get('models', [])
        if models:
            for model in models:
                print(f"  - {model.get('name', 'unknown')}")
        else:
            print("  (No models installed)")
            print()
            print("To install a model, run:")
            print("  ollama pull llama2")
    else:
        print(f"✗ Server responded with status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Ollama server is not running")
    print()
    print("Starting Ollama server...")
    try:
        # Try to start server
        import subprocess
        process = subprocess.Popen(['ollama', 'serve'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
        print("  Server process started (PID: {})".format(process.pid))
        print("  Waiting for server to initialize...")
        import time
        time.sleep(3)
        
        # Check again
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                print("✓ Server started successfully!")
            else:
                print("✗ Server started but not responding correctly")
        except:
            print("✗ Server may still be starting. Please wait a few seconds and try again.")
            print("  Or run manually: ollama serve")
    except Exception as e:
        print(f"✗ Failed to start server: {e}")
        print()
        print("Please start Ollama server manually:")
        print("  1. Open a new terminal")
        print("  2. Run: ollama serve")
        print("  3. Keep that terminal open")
except Exception as e:
    print(f"✗ Error connecting to server: {e}")

print()
print("=" * 60)

