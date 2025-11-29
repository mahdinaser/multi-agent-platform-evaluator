# Multi-LLM Model Testing

## Overview

The framework now supports testing multiple LLM models simultaneously to determine which models are best at database platform selection.

## Supported Models

### Models to Pull

```bash
# Pull all test models
ollama pull qwen3:14b
ollama pull deepseek-r1:14b
ollama pull llama3.1:70b    # Requires good GPU (40GB+ VRAM)
ollama pull qwen2.5:14b
ollama pull llama2:latest   # Already have this
```

### Model Specifications

| Model | Parameters | Context Window | Strengths |
|-------|-----------|---------------|-----------|
| **llama2:latest** | 7B | 4K | General purpose, fast |
| **qwen3:14b** | 14B | 32K | Strong reasoning, multilingual |
| **deepseek-r1:14b** | 14B | 16K | Code & reasoning optimized |
| **llama3.1:70b** | 70B | 128K | Best performance, slow |
| **qwen2.5:14b** | 14B | 32K | Latest Qwen, improved |

## Configuration

Enable multi-model testing in `config.yaml`:

```yaml
llm_config:
  use_local: true
  use_ollama: true
  model_name: "llama2"  # Default/baseline model
  
  # Multiple LLM models to test
  test_models:
    - "llama2:latest"
    - "qwen3:14b"
    - "deepseek-r1:14b"
    - "llama3.1:70b"
    - "qwen2.5:14b"
  
  # Enable multi-model testing
  enable_multi_model: true
```

## Generated Agents

When `enable_multi_model: true`, the framework automatically creates:

### Standard LLM Agents (5)
- `llm_llama2_latest` - Baseline
- `llm_qwen3_14b`
- `llm_deepseek_r1_14b`
- `llm_llama3_1_70b`
- `llm_qwen2_5_14b`

### MCP-Enhanced LLM Agents (5)
- `llm_mcp_llama2_latest`
- `llm_mcp_qwen3_14b`
- `llm_mcp_deepseek_r1_14b`
- `llm_mcp_llama3_1_70b`
- `llm_mcp_qwen2_5_14b`

**Total:** 10 LLM agent variants (5 standard + 5 MCP)

## Research Questions

### Q1: Which LLM is best for system optimization?
**Compare:** All 5 LLM models
- Measure: Mean latency of selected platforms
- Hypothesis: Larger models (70B) outperform smaller (7B, 14B)
- Reality check: Does 70B justify 10x slower inference?

### Q2: Does model size matter?
**Compare:** 7B (llama2) vs 14B (qwen3, deepseek) vs 70B (llama3.1)
- Track: Decision quality vs decision time
- Find: Sweet spot for production deployment

### Q3: Specialized models vs general-purpose?
**Compare:** deepseek-r1 (code-optimized) vs llama2 (general)
- Hypothesis: Code-specialized models understand system trade-offs better
- Test: Do they select better platforms for technical queries?

### Q4: Impact of context window?
**Compare:** llama2 (4K) vs qwen3 (32K) vs llama3.1 (128K)
- With long performance histories, do larger context windows help?
- Test: MCP agents with extensive tool output

### Q5: Does MCP benefit all models equally?
**Compare:** Each model with/without MCP
- Measure: Improvement = LLM+MCP latency - LLM latency
- Hypothesis: Smaller models benefit more from tool access

### Q6: Model consistency?
**Track:** Decision variance across models
- Do all models agree on best platform for simple cases?
- Where do models disagree? (reveals interesting edge cases)

## Expected Results

### Performance Hierarchy (Predicted)

```
Platform Selection Quality
     ↑
     │  llama3.1:70b + MCP  ← Best (but slowest)
     │  ════════════════════
     │  qwen3:14b + MCP
     │  qwen2.5:14b + MCP
     │  deepseek-r1:14b + MCP
     │  ────────────────────
     │  llama3.1:70b (no MCP)
     │  ────────────────────
     │  qwen3:14b
     │  qwen2.5:14b
     │  deepseek-r1:14b
     │  ────────────────────
     │  llama2:latest + MCP
     │  ────────────────────
     │  llama2:latest
     │  ════════════════════
     └──────────────────────→ Time
```

### Decision Time (Predicted)

```
Decision Latency (ms)
     ↑
     │  llama3.1:70b    ← 5000-10000ms
     │  ════════════════
     │  qwen3:14b       ← 1000-2000ms
     │  qwen2.5:14b     ← 1000-2000ms
     │  deepseek-r1:14b ← 1000-2000ms
     │  ────────────────
     │  llama2:latest   ← 500-1000ms
     │  ════════════════
     └──────────────────→ Time
```

## Analysis Tables

The framework will generate:

### 1. **LLM Model Comparison Table**
| Model | Mean Latency | Decision Time | Accuracy | Regret |
|-------|-------------|---------------|----------|--------|
| llama3.1:70b | ? | ? | ? | ? |
| qwen3:14b | ? | ? | ? | ? |
| ... | ... | ... | ... | ... |

### 2. **MCP Impact by Model**
| Model | No MCP | With MCP | Improvement | MCP Benefit % |
|-------|--------|----------|-------------|---------------|
| llama2 | ? | ? | ? | ? |
| qwen3 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... |

### 3. **Model Characteristics**
| Model | Size | Context | Speed | Quality | Best Use Case |
|-------|------|---------|-------|---------|---------------|
| llama3.1:70b | 70B | 128K | Slow | Excellent | Offline/batch |
| qwen3:14b | 14B | 32K | Medium | Very Good | Production |
| deepseek-r1:14b | 14B | 16K | Medium | Very Good | Code tasks |
| qwen2.5:14b | 14B | 32K | Medium | Very Good | General |
| llama2:latest | 7B | 4K | Fast | Good | Real-time |

## Plots

### 1. **LLM Model Comparison (Bar Chart)**
- X-axis: LLM models
- Y-axis: Mean latency of selected platforms
- Grouped bars: with/without MCP

### 2. **Decision Time vs Quality (Scatter)**
- X-axis: Decision time (ms)
- Y-axis: Platform selection quality
- Points: Each model
- Pareto frontier: Best quality/time trade-offs

### 3. **MCP Benefit Heatmap**
- Rows: LLM models
- Columns: Experiment types (scan, filter, aggregate, ...)
- Values: MCP improvement (color-coded)

### 4. **Model Agreement Matrix**
- Pairwise comparison: How often do models agree?
- Identify consensus vs controversial cases

## Running Experiments

### Quick Test (Subset of Models)
```yaml
llm_config:
  enable_multi_model: true
  test_models:
    - "llama2:latest"
    - "qwen3:14b"
```

### Full Test (All Models)
```yaml
llm_config:
  enable_multi_model: true
  test_models:
    - "llama2:latest"
    - "qwen3:14b"
    - "deepseek-r1:14b"
    - "llama3.1:70b"
    - "qwen2.5:14b"
```

### Minimal Test (Just One)
```yaml
llm_config:
  enable_multi_model: false
  model_name: "qwen3:14b"  # Just test this one
```

## Performance Considerations

### GPU Requirements

| Models Tested | VRAM Needed | Recommendation |
|---------------|-------------|----------------|
| 1 model (7B/14B) | 8-12 GB | RTX 3060, 4060 Ti |
| 2-3 models (14B) | 16-24 GB | RTX 4070 Ti, 4080 |
| llama3.1:70b | 40+ GB | RTX 4090, A100 |

**Note:** Models are loaded sequentially, not simultaneously, so VRAM is per-model.

### Execution Time

**Rough estimates:**

| Configuration | Experiments | Time per Model | Total Time |
|--------------|-------------|----------------|------------|
| Quick mode, 1 model | ~50 | ~30 min | ~30 min |
| Quick mode, 5 models | ~250 | ~30 min each | ~2.5 hours |
| Full mode, 5 models | ~1000 | ~2 hours each | ~10 hours |

**Tip:** Start with `quick_mode: true` and 2-3 models to test the pipeline.

## Paper Contribution

This multi-LLM testing adds a valuable research dimension:

### Section: LLM Model Comparison
**Novel insights:**
- First systematic comparison of LLMs for system optimization
- Quantifies model size vs quality trade-off
- Shows MCP benefit varies by model
- Identifies which models are best for database selection

### Key Findings (Expected)
1. **70B models are best but impractical** (10x slower)
2. **14B models are sweet spot** (good quality, reasonable speed)
3. **MCP helps all models** but smaller models benefit more
4. **Code-specialized models** (deepseek) perform well
5. **Context window matters** for MCP agents with long history

## Implementation Details

**Agent Naming Convention:**
```
llm_{model}_{size}           # Standard LLM
llm_mcp_{model}_{size}       # With MCP tools

Examples:
- llm_qwen3_14b              # Qwen3 14B standard
- llm_mcp_deepseek_r1_14b    # DeepSeek-R1 with MCP
```

**Model Detection:**
- Framework checks `ollama list` for available models
- Gracefully skips models that aren't pulled
- Logs warnings for missing models

**Fallback Behavior:**
- If model not found → try default (llama2)
- If Ollama not running → use simple reasoning
- Experiments continue even if some models fail

## Quick Start

```bash
# 1. Pull models
ollama pull qwen3:14b
ollama pull deepseek-r1:14b

# 2. Update config.yaml
# Set enable_multi_model: true

# 3. Run experiments
cd F:\workspace\PVLDB\multi_agent_platform_evaluator
python app.py

# 4. Check results
# Look for:
#   - summary_llm_models.csv
#   - llm_model_comparison.png
#   - mcp_impact_by_model.csv
```

## Status

- ✅ Configuration support added
- ✅ Multi-model agent initialization
- ✅ Automatic agent naming
- ✅ Graceful model detection and fallback
- 🚧 Pending: LLM-specific analysis tables
- 🚧 Pending: LLM comparison plots
- 🚧 Pending: Paper section on LLM comparison

## References

- Qwen: https://github.com/QwenLM/Qwen
- DeepSeek: https://github.com/deepseek-ai/DeepSeek-Coder
- Llama: https://github.com/meta-llama/llama3
- Ollama: https://ollama.ai/library

