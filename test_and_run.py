#!/usr/bin/env python3
"""Test script to verify environment and run app with monitoring."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Environment Check")
print("=" * 60)

# Check Python version
print(f"Python version: {sys.version}")

# Check Ollama
print("\nChecking Ollama...")
try:
    import ollama
    try:
        models_response = ollama.list()
        models = [m['name'] for m in models_response.get('models', [])]
        print(f"✓ Ollama is running!")
        print(f"  Available models: {models}")
        
        # Check if llama2 is available
        llama2_found = any('llama2' in m.lower() for m in models)
        if llama2_found:
            print(f"  ✓ llama2 model found!")
        else:
            print(f"  ⚠ llama2 not found. Available models: {models}")
            if models:
                print(f"  → Will use: {models[0]}")
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print("  Make sure Ollama server is running: ollama serve")
except ImportError:
    print("✗ ollama package not installed")
    print("  Install with: pip install ollama")

# Check config
print("\nChecking configuration...")
try:
    from src.utils import load_config
    config_path = project_root / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    llm_config = config.get('llm_config', {})
    print(f"✓ Config loaded")
    print(f"  LLM config: use_ollama={llm_config.get('use_ollama')}, model_name={llm_config.get('model_name')}")
except Exception as e:
    print(f"✗ Config error: {e}")

# Try importing main modules
print("\nChecking imports...")
try:
    from src.data_generator import DataGenerator
    print("✓ DataGenerator")
except Exception as e:
    print(f"✗ DataGenerator: {e}")

try:
    from src.experiment_runner import ExperimentRunner
    print("✓ ExperimentRunner")
except Exception as e:
    print(f"✗ ExperimentRunner: {e}")

try:
    from src.agent_manager import AgentManager
    print("✓ AgentManager")
except Exception as e:
    print(f"✗ AgentManager: {e}")

try:
    from agents.agent_llm import LLMAgent
    print("✓ LLMAgent")
except Exception as e:
    print(f"✗ LLMAgent: {e}")

print("\n" + "=" * 60)
print("Starting app.py...")
print("=" * 60)
print()

# Now run the app
try:
    import app
    # The app should run when imported if it has if __name__ == "__main__"
    # Otherwise, we'll call main()
    if hasattr(app, 'main'):
        app.main()
    else:
        print("App module loaded but no main() function found")
except Exception as e:
    print(f"Error running app: {e}")
    import traceback
    traceback.print_exc()

