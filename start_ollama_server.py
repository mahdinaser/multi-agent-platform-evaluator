#!/usr/bin/env python3
"""
Python script to start and manage Ollama server.
"""
import subprocess
import time
import sys
import os
import requests
from pathlib import Path

def find_ollama():
    """Find Ollama executable."""
    # Common locations
    locations = [
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe'),
        os.path.join('C:', 'Program Files', 'Ollama', 'ollama.exe'),
        os.path.join('C:', 'Program Files (x86)', 'Ollama', 'ollama.exe'),
    ]
    
    # Check PATH
    try:
        result = subprocess.run(['where', 'ollama'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    # Check common locations
    for loc in locations:
        if os.path.exists(loc):
            return loc
    
    return None

def check_server_running():
    """Check if Ollama server is already running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        return True, response
    except:
        return False, None

def start_ollama_server(ollama_path):
    """Start Ollama server."""
    print("Starting Ollama server...")
    print(f"Using: {ollama_path}")
    
    # Start server in background
    try:
        process = subprocess.Popen(
            [ollama_path, 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        print(f"✓ Ollama server process started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"✗ Failed to start server: {e}")
        return None

def main():
    print("=" * 60)
    print("Ollama Server Manager")
    print("=" * 60)
    print()
    
    # Check if server is already running
    is_running, response = check_server_running()
    if is_running:
        print("✓ Ollama server is already running!")
        print()
        if response:
            try:
                models = response.json().get('models', [])
                if models:
                    print("Available models:")
                    for model in models:
                        print(f"  - {model.get('name', 'unknown')}")
                else:
                    print("No models installed yet.")
            except:
                pass
        return 0
    
    # Find Ollama
    print("Looking for Ollama installation...")
    ollama_path = find_ollama()
    
    if not ollama_path:
        print("✗ Ollama is not installed or not found!")
        print()
        print("Please install Ollama:")
        print("  1. Download from: https://ollama.ai/download")
        print("  2. Install the application")
        print("  3. Restart your terminal")
        print("  4. Run this script again")
        return 1
    
    print(f"✓ Found Ollama at: {ollama_path}")
    print()
    
    # Start server
    process = start_ollama_server(ollama_path)
    if not process:
        return 1
    
    # Wait and verify
    print("Waiting for server to start...")
    for i in range(10):
        time.sleep(1)
        is_running, response = check_server_running()
        if is_running:
            print("✓ Ollama server is running!")
            print()
            print("Server is ready on http://localhost:11434")
            print()
            print("To stop the server, close this window or run:")
            print(f"  taskkill /F /PID {process.pid}")
            print()
            return 0
        print(f"  Waiting... ({i+1}/10)")
    
    print("⚠ Server may still be starting. Please wait a few more seconds.")
    print(f"Process ID: {process.pid}")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

