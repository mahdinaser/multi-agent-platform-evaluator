# Model Context Protocol (MCP) Integration

## Overview

MCP (Model Context Protocol) integration allows the LLM agent to access tools and historical data for making more informed platform selection decisions.

## What is MCP?

Model Context Protocol is a standardized way for LLMs to interact with external tools and data sources. Instead of relying solely on pre-trained knowledge, MCP-enabled LLMs can:
- Query databases
- Access real-time performance metrics
- Call functions to retrieve context
- Use tools to gather information before making decisions

## MCP Agent vs Standard LLM Agent

### Standard LLM Agent (`agent_llm.py`)
- **Input:** Data source name, experiment type, available platforms
- **Knowledge:** Only pre-trained model knowledge + prompt description
- **Decision:** Based on general understanding of platforms
- **Limitations:** Cannot access actual performance history

### MCP-Enhanced LLM Agent (`agent_llm_mcp.py`)
- **Input:** Same as standard
- **Tools Available:**
  1. `get_performance_history(platform, data_source, experiment_type)` - Historical latencies
  2. `compare_platforms(platforms, data_source, experiment_type)` - Side-by-side comparison
  3. `get_platform_specs(platform)` - Platform characteristics
- **Decision:** Based on actual performance data + platform specs
- **Advantages:** Data-driven decisions, learns from history

## MCP Tools Implemented

### 1. Performance History Tool
```python
get_performance_history(platform, data_source, experiment_type)
→ {
    'mean_latency': 23.5,
    'std_latency': 5.2,
    'min_latency': 15.0,
    'max_latency': 45.0,
    'count': 10
}
```

### 2. Platform Comparison Tool
```python
compare_platforms(['pandas', 'duckdb', 'polars'], 'tabular_1000000', 'aggregate')
→ [
    {'platform': 'duckdb', 'mean_latency': 12.3, 'std_latency': 2.1},
    {'platform': 'polars', 'mean_latency': 15.7, 'std_latency': 3.4},
    {'platform': 'pandas', 'mean_latency': 45.2, 'std_latency': 8.9}
]
```

### 3. Platform Specifications Tool
```python
get_platform_specs('duckdb')
→ {
    'type': 'analytical SQL engine',
    'strengths': ['columnar storage', 'fast aggregations'],
    'best_for': ['large datasets', 'OLAP queries'],
    'limitations': ['setup overhead', 'not for OLTP']
}
```

## Example MCP-Enhanced Prompt

```
You are an expert database system selector. Select the best platform for this workload.

TASK:
- Data Source: tabular_1000000
- Experiment Type: aggregate
- Available Platforms: pandas, duckdb, polars

HISTORICAL PERFORMANCE DATA (via MCP tools):
Platform performance for similar workloads:
  - duckdb: 12.30ms avg (±2.10ms, n=8)
  - polars: 15.70ms avg (±3.40ms, n=8)
  - pandas: 45.20ms avg (±8.90ms, n=8)

PLATFORM SPECIFICATIONS (via MCP tools):

DUCKDB:
  Type: analytical SQL engine
  Best for: large datasets, OLAP queries, aggregations

POLARS:
  Type: fast DataFrame library
  Best for: large datasets, complex transformations, performance-critical

PANDAS:
  Type: in-memory DataFrame
  Best for: small-medium data, mixed operations, prototyping

Based on the historical performance data and platform specifications, 
which platform should be used for aggregate on tabular_1000000?
```

## Expected Impact on Performance

### Hypothesis
MCP-enhanced LLM should outperform standard LLM because:
1. **Data-driven decisions:** Access to actual performance metrics
2. **Adaptive learning:** Performance history grows over time
3. **Context-aware:** Can compare platforms for specific workload combinations
4. **Informed reasoning:** Platform specs + historical data

### Metrics to Compare

| Metric | Standard LLM | MCP-Enhanced LLM | Expected |
|--------|-------------|------------------|----------|
| Mean Latency | 21.20 ms | ? | **Lower** |
| Decision Accuracy | 100% | ? | **Same or higher** |
| Decision Time | ~50s | ? | **Similar** (extra tool calls minimal) |
| Adaptation Rate | N/A | ? | **Improves over time** |

### Experimental Design

Run experiments comparing:
1. **LLM (no MCP)**: Current implementation
2. **LLM + MCP**: New implementation with tool access
3. **Measure:**
   - Latency of selected platforms
   - Decision accuracy (vs Oracle)
   - Performance improvement over time
   - Tool usage statistics

## Implementation Status

- [x] MCP agent implementation (`agent_llm_mcp.py`)
- [x] Three MCP tools implemented
- [x] Tool result integration in prompts
- [x] Performance history tracking
- [x] Statistics collection
- [ ] Integration with agent_manager
- [ ] Comparison experiments (LLM vs LLM+MCP)
- [ ] MCP-specific analysis plots
- [ ] Paper section on MCP results

## Usage

Add to `config.yaml`:
```yaml
agents:
  - llm       # Standard LLM (no tools)
  - llm_mcp   # LLM with MCP tool access
```

The framework will run both and generate comparative analysis showing:
- LLM decision quality with vs without tool access
- Impact of historical data on decision-making
- Tool usage patterns
- Performance improvement curves

## Research Questions

1. **Does tool access improve LLM platform selection?**
   - Compare latency: LLM vs LLM+MCP
   - Measure decision accuracy: both agents vs Oracle

2. **How much historical data is needed?**
   - Plot MCP agent performance vs. historical data coverage
   - Identify minimum samples for reliable decisions

3. **Which tools are most valuable?**
   - Track tool usage frequency
   - Correlation between tool use and decision quality

4. **Does MCP overhead justify the benefit?**
   - Measure decision time increase
   - Calculate performance/overhead trade-off

## Expected Paper Contribution

This adds a novel research dimension:
- **First study** of MCP for database platform selection
- **Empirical validation** of tool-augmented LLMs for system optimization
- **Comparison**: LLM (knowledge-based) vs LLM+MCP (data-driven)
- **Insights** on when tool access helps vs when model knowledge suffices

## References

- Model Context Protocol: https://modelcontextprotocol.io/
- Anthropic MCP Announcement: https://www.anthropic.com/news/model-context-protocol
- MCP Specification: https://spec.modelcontextprotocol.io/

