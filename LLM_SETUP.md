# LLM Agent Setup Guide

## Overview

The LLM agent now uses **actual language models** instead of simulation. You can choose between:

1. **Ollama** (Recommended - Easiest)
   - Local LLM runner with many models
   - Easy installation and usage
   - No API costs
   - Fast inference
   - Supports many models (Llama2, Mistral, CodeLlama, etc.)

2. **Local Transformers** (Alternative)
   - Uses Hugging Face Transformers
   - Runs entirely locally
   - No API costs
   - More setup required

3. **API-based Models** (Optional)
   - Uses OpenAI API
   - Requires API key
   - Faster but costs money
   - Requires internet connection

## Configuration

Edit `config/config.yaml`:

```yaml
llm_config:
  use_local: true  # true for local, false for API
  use_ollama: true  # Use Ollama (recommended), falls back to transformers if false
  model_name: "llama2"  # Ollama model name or transformers model
  api_key: null  # Set OPENAI_API_KEY env var if using API
```

## Ollama Setup (Recommended)

### Step 1: Install Ollama
Download and install from: https://ollama.ai/

### Step 2: Pull a Model
```bash
# Recommended models:
ollama pull llama2          # General purpose (3.8GB)
ollama pull mistral         # Fast and efficient (4.1GB)
ollama pull codellama       # Code-focused (3.8GB)
ollama pull llama2:7b       # Smaller version (3.8GB)
ollama pull llama2:13b      # Larger, better quality (7.3GB)
```

### Step 3: Install Python Package
```bash
pip install ollama
```

### Step 4: Verify Ollama is Running
```bash
ollama list  # Should show your installed models
```

### Step 5: Update Config
```yaml
llm_config:
  use_local: true
  use_ollama: true
  model_name: "llama2"  # Use the model you pulled
```

## Local Transformers Setup (Alternative)

### Option 1: Small Model (Fast, Lower Quality)
```yaml
model_name: "microsoft/DialoGPT-small"
```
- Fast inference
- Lower memory usage (~100MB)
- Good for testing

### Option 2: Better Model (Slower, Higher Quality)
```yaml
model_name: "gpt2"
```
- Better reasoning
- Still relatively fast
- ~500MB memory

### Option 3: Best Local Model (Slowest, Best Quality)
```yaml
model_name: "microsoft/DialoGPT-medium"
```
- Best reasoning quality
- Slower inference
- ~1GB memory

## API Setup (OpenAI)

1. Get API key from https://platform.openai.com/
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
3. Update config:
   ```yaml
   llm_config:
     use_local: false
     model_name: "gpt-3.5-turbo"
   ```

## Installation

### For Ollama (Recommended):
```bash
# 1. Install Ollama from https://ollama.ai/
# 2. Pull a model: ollama pull llama2
# 3. Install Python package:
pip install ollama
```

### For Local Transformers (Alternative):
```bash
pip install transformers torch
```

### For API (Optional):
```bash
pip install openai
```

## Performance Notes

- **Ollama**: Fastest local option, ~0.5-2 seconds per decision
- **Local transformers**: ~1-5 seconds per decision
- **API inference**: ~0.5-2 seconds per decision
- **Fallback**: If LLM fails, uses simple rule-based reasoning

## Recommended Models

### For Ollama:
- **llama2** - Best balance (3.8GB)
- **mistral** - Fast and efficient (4.1GB)
- **codellama** - Good for technical decisions (3.8GB)
- **llama2:7b** - Smaller, faster (3.8GB)
- **llama2:13b** - Better quality, slower (7.3GB)

### For Transformers:
- **microsoft/DialoGPT-small** - Small, fast (~100MB)
- **gpt2** - Good balance (~500MB)
- **microsoft/DialoGPT-medium** - Better quality (~500MB)

## Troubleshooting

### Model Download Issues
- First run downloads model automatically
- May take several minutes depending on internet speed
- Models cached in `~/.cache/huggingface/`

### Out of Memory
- Use smaller model: `"gpt2"` or `"microsoft/DialoGPT-small"`
- Reduce batch size in code if needed

### API Errors
- Check API key is set correctly
- Verify internet connection
- Check API quota/balance

## Fallback Behavior

If LLM initialization fails, the agent automatically falls back to simple rule-based reasoning. This ensures the experiment always runs, even without LLM dependencies.

