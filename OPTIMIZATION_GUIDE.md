# Performance Optimization Guide

## Optimizations Applied

### 1. **Data Caching** ✅
- Data is now loaded once and cached in memory
- Eliminates repeated file I/O operations
- **Speedup: ~10-50x** for experiments with same data source

### 2. **Quick Mode** ✅
- New `quick_mode` option in config.yaml
- When enabled, only runs experiments on agent-selected platforms
- **Speedup: ~6x** (5 agents × 1 platform vs 5 agents × 6 platforms)

### 3. **Reduced Data Sizes** ✅
- Default data sizes reduced in config.yaml:
  - Logs: 1M → 100K events
  - Vectors: 100K → 10K vectors
  - Time-series: 1M → 100K samples
  - Text: 50K → 5K documents
- **Speedup: ~5-10x** depending on operation

### 4. **Pre-loading** ✅
- All data sources loaded once at start
- Avoids repeated loading during experiments
- **Speedup: ~2-3x**

## Configuration Options

### Quick Mode (Fastest)
Edit `config/config.yaml`:
```yaml
experiment:
  quick_mode: true  # Only test agent-selected platforms
```

**Result:** ~280 experiments instead of ~1,680 (6x faster)

### Reduced Data Sizes
Already configured in `config/config.yaml`:
```yaml
data:
  logs:
    num_events: 100000  # Reduced from 1000000
  vectors:
    num_vectors: 10000  # Reduced from 100000
```

### Limit Tabular Sizes
```yaml
experiment:
  max_data_size: 500000  # Skip larger datasets
```

### Minimal Testing (Fastest)
For quick testing, use:
```yaml
data:
  tabular:
    sizes: [50000]  # Only smallest size

experiment:
  quick_mode: true
```

**Result:** ~35 experiments (50x faster than full run)

## Performance Comparison

| Mode | Experiments | Estimated Time | Use Case |
|------|------------|----------------|----------|
| **Full** | ~1,680 | 2-4 hours | Complete evaluation |
| **Quick** | ~280 | 20-40 min | Agent evaluation |
| **Minimal** | ~35 | 3-5 min | Quick testing |

## Additional Tips

1. **Skip Unnecessary Platforms**: Remove platforms you don't need from `config.yaml`
2. **Reduce Experiment Types**: Comment out experiment types you don't need
3. **Use Fewer Agents**: Test with 1-2 agents first, then add more
4. **Parallel Execution**: Future enhancement (currently sequential)

## Monitoring Progress

The progress bar shows:
- Current experiment number
- Total experiments
- Estimated time remaining (if tqdm supports it)

## Troubleshooting

If still slow:
1. Check data sizes in config.yaml
2. Enable quick_mode
3. Reduce number of agents/platforms
4. Use smaller tabular datasets only

