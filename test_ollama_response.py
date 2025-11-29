"""Test what ollama.list() actually returns."""
import ollama
import json

try:
    response = ollama.list()
    print("=" * 60)
    print("Ollama.list() Response:")
    print("=" * 60)
    print(f"Type: {type(response)}")
    print(f"Response: {response}")
    print()
    
    if isinstance(response, dict):
        print("Keys:", list(response.keys()))
        if 'models' in response:
            print(f"Models list: {response['models']}")
            print(f"Number of models: {len(response['models'])}")
            if response['models']:
                print(f"First model: {response['models'][0]}")
                print(f"First model type: {type(response['models'][0])}")
                if isinstance(response['models'][0], dict):
                    print(f"First model keys: {list(response['models'][0].keys())}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

