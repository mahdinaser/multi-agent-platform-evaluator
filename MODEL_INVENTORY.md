# LLM Model Inventory

## Models Being Tested (7 models → 14 agents)

### ✅ LLAMA FAMILY (3 models)
| Model | Size | Parameters | Use Case |
|-------|------|------------|----------|
| `llama2:latest` | 3.8 GB | 7B | Original baseline |
| `llama3:latest` | 4.7 GB | 8B | Improved version |
| `llama3.1:70b` | 42 GB | 70B | Best quality |

**Research Question:** How much does Llama improve from v2 → v3 → v3.1?

### ✅ QWEN FAMILY (3 models)
| Model | Size | Parameters | Context | Use Case |
|-------|------|------------|---------|----------|
| `qwen3:latest` | 5.2 GB | ~7B | 32K | NEW! Smaller Qwen3 |
| `qwen3:14b` | 9.3 GB | 14B | 32K | Full Qwen3 |
| `qwen2.5:14b` | 9.0 GB | 14B | 32K | Previous gen |

**Research Questions:**
- Is Qwen 3 better than Qwen 2.5?
- Does the smaller qwen3:latest compete with full 14B models?
- Does larger context (32K) help?

### ✅ DEEPSEEK (1 model)
| Model | Size | Parameters | Specialization |
|-------|------|------------|----------------|
| `deepseek-r1:14b` | 9.0 GB | 14B | Code & reasoning |

**Research Question:** Do code-specialized models outperform general-purpose for DB selection?

---

## Framework Will Create 14 Agents

### Standard LLM Agents (7)
1. `llm_llama2_latest`
2. `llm_llama3_latest`
3. `llm_llama3_1_70b`
4. `llm_qwen3_latest` ← NEW!
5. `llm_qwen3_14b`
6. `llm_qwen2_5_14b`
7. `llm_deepseek_r1_14b`

### MCP-Enhanced Agents (7)
1. `llm_mcp_llama2_latest`
2. `llm_mcp_llama3_latest`
3. `llm_mcp_llama3_1_70b`
4. `llm_mcp_qwen3_latest` ← NEW!
5. `llm_mcp_qwen3_14b`
6. `llm_mcp_qwen2_5_14b`
7. `llm_mcp_deepseek_r1_14b`

---

## Additional Models Available (Optional)

### Not Included in Current Tests
| Model | Size | Parameters | Why Not Included |
|-------|------|------------|------------------|
| `deepseek-r1:8b` | 5.2 GB | 8B | Smaller variant (can add) |
| `qwen3-vl:30b` | 19 GB | 30B | Vision model (not needed for DB selection) |

**You can add these by uncommenting in `config.yaml`!**

---

## Model Size Distribution

```
Size Range          Models  Total Size
─────────────────────────────────────
Small (3-5 GB)      3       13.5 GB
  - llama2          3.8 GB
  - llama3          4.7 GB
  - qwen3:latest    5.2 GB

Medium (9 GB)       3       27.0 GB
  - qwen2.5:14b     9.0 GB
  - qwen3:14b       9.3 GB
  - deepseek:14b    9.0 GB

Large (42 GB)       1       42.0 GB
  - llama3.1:70b    42.0 GB

TOTAL              7       82.5 GB
```

---

## Research Comparisons Enabled

### 1. Model Evolution
- **Llama:** v2 → v3 → v3.1 (7B → 8B → 70B)
- **Qwen:** v2.5 → v3 (14B → 14B → 7B variants)

### 2. Size vs Quality
- Small: 3.8-5.2 GB (3 models)
- Medium: 9.0-9.3 GB (3 models)
- Large: 42 GB (1 model)

### 3. Architecture Comparison
- Llama architecture (Meta)
- Qwen architecture (Alibaba)
- DeepSeek architecture (DeepSeek AI)

### 4. Specialization
- General purpose: Llama, Qwen
- Code-optimized: DeepSeek

### 5. MCP Tool Benefit
- Each model tested with/without tools
- Which models benefit most from MCP?

### 6. Context Window Impact
- Small: 4K (llama2)
- Large: 32K (qwen family)
- Huge: 128K (llama3.1)

---

## Expected Performance Hierarchy

```
Platform Selection Quality
     ↑
     │  llama3.1:70b + MCP        ← Best (slowest)
     │  ═══════════════════
     │  qwen3:14b + MCP
     │  qwen2.5:14b + MCP
     │  deepseek-r1:14b + MCP
     │  ───────────────────
     │  llama3:latest + MCP
     │  qwen3:latest + MCP
     │  ───────────────────
     │  llama2:latest + MCP
     │  ═══════════════════
     │  (no MCP variants...)
     └─────────────────────→ Decision Time
                              Faster
```

---

## Quick Commands

### Check Available Models
```bash
ollama list
```

### Pull Additional Models
```bash
ollama pull deepseek-r1:8b
```

### Test Specific Model
```bash
# Edit config.yaml and set:
llm_config:
  enable_multi_model: false
  model_name: "qwen3:14b"  # Test just this one
```

### Test All Models
```bash
# Edit config.yaml and set:
llm_config:
  enable_multi_model: true
  # All models in test_models list will be tested
```

---

## Performance Estimates

| Configuration | Agents | Experiments | Estimated Time |
|---------------|--------|-------------|----------------|
| Quick test, 1 model | 2 | ~50 | ~30 min |
| Quick test, 7 models | 14 | ~350 | ~3-4 hours |
| Full test, 7 models | 14 | ~1400 | ~14-20 hours |

**Note:** llama3.1:70b is much slower than others (10x). Consider running quick tests first!

---

## Status: ✅ READY

- **Models Downloaded:** 9 total (7 will be tested)
- **Disk Space Used:** ~95 GB
- **Agents Configured:** 14 (7 standard + 7 MCP)
- **Framework Ready:** Yes
- **Multi-Model Testing:** Enabled

Run `python app.py` to start experiments!

