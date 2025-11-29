"""Quick script to check Ollama and pull a model if needed."""
import sys

try:
    import ollama
    
    # Check available models
    try:
        models_response = ollama.list()
        models = [m['name'] for m in models_response.get('models', [])]
        print(f"Available Ollama models: {models}")
        
        if not models:
            print("No models found. Pulling 'phi' (small, fast model)...")
            ollama.pull('phi')
            print("Model 'phi' pulled successfully!")
        elif 'llama2' in models or any('llama2' in m for m in models):
            print("llama2 model found!")
        elif 'phi' in models or any('phi' in m for m in models):
            print("phi model found!")
        else:
            print(f"Using available model: {models[0]}")
    except Exception as e:
        print(f"Error checking models: {e}")
        print("Trying to pull 'phi' model...")
        try:
            ollama.pull('phi')
            print("Model 'phi' pulled successfully!")
        except Exception as e2:
            print(f"Error pulling model: {e2}")
            sys.exit(1)
            
except ImportError:
    print("ollama package not installed. Install with: pip install ollama")
    sys.exit(1)
except Exception as e:
    print(f"Ollama not available: {e}")
    print("Make sure Ollama is installed and running.")
    print("Download from: https://ollama.ai/download")
    sys.exit(1)


