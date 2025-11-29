# Enhanced Analysis Features

## New Analysis Tables (6 additional)

### 1. **Performance Rankings** (`summary_performance_rankings.csv`)
- Ranks platforms and agents by each performance metric
- Shows which platform/agent is best for latency, throughput, memory, CPU
- Includes direction indicators (lower_better vs higher_better)

### 2. **Correlations** (`summary_correlations.csv`)
- Correlation matrix between all metrics
- Identifies relationships (e.g., latency vs memory trade-offs)
- Helps understand metric dependencies

### 3. **Agent Effectiveness** (`summary_agent_effectiveness.csv`)
- Agent performance broken down by data source and experiment type
- Shows efficiency ratio (how close to optimal)
- Identifies which agents excel in which scenarios

### 4. **Platform Recommendations** (`summary_platform_recommendations.csv`)
- Best platform for each scenario (data source × experiment type)
- Recommendations for:
  - Lowest latency
  - Highest throughput
  - Lowest memory usage
- Actionable insights for platform selection

### 5. **Statistical Tests** (`summary_statistical_tests.csv`)
- Mann-Whitney U tests comparing platforms and agents
- P-values and significance indicators
- Identifies statistically significant performance differences

### 6. **Cost-Benefit Analysis** (`summary_cost_benefit.csv`)
- Efficiency scores combining latency, memory, and CPU
- Latency-per-memory ratios
- Helps identify most efficient platforms overall

## New Visualizations (8 additional)

### 1. **Platform-Experiment Heatmap**
- Shows latency for each platform × experiment type combination
- Quickly identifies best platforms for specific operations

### 2. **Agent Learning Curves**
- Tracks agent performance over time
- Shows latency and success rate trends
- Identifies which agents improve with experience

### 3. **Latency vs Throughput Scatter**
- Trade-off visualization
- Shows which platforms offer best balance
- Log scale for better visibility

### 4. **Platform Efficiency**
- Efficiency scores (latency per unit memory)
- 3D scatter showing latency, memory, CPU trade-offs
- Bubble size represents CPU time

### 5. **Agent Decision Patterns**
- 4-panel analysis:
  - Platform selection frequency by agent
  - Platform selection by data source
  - Platform selection by experiment type
  - Agent decision consistency scores

### 6. **Performance Distributions (Violin Plots)**
- Shows full distribution of metrics, not just averages
- Reveals variability and outliers
- Separate plots for latency, memory, throughput

### 7. **Correlation Heatmap**
- Visual correlation matrix
- Color-coded relationships between metrics
- Identifies strong positive/negative correlations

### 8. **Best Platform Scenarios**
- Matrix showing best platform for each scenario
- Quick reference for platform selection
- Color-coded by platform

## Total Analysis Output

### Summary Tables: **18 tables**
1. Overall summary
2. By platform
3. By agent
4. By data source
5. Latency statistics
6. Memory statistics
7. CPU statistics
8. Vector performance
9. Text similarity
10. Stability
11. Agent accuracy
12. Agent regret
13. **Performance rankings** (NEW)
14. **Correlations** (NEW)
15. **Agent effectiveness** (NEW)
16. **Platform recommendations** (NEW)
17. **Statistical tests** (NEW)
18. **Cost-benefit analysis** (NEW)

### Visualizations: **18 plots**
1. Latency distribution
2. Memory comparison
3. CPU comparison
4. Stability heatmap
5. Agent decisions
6. Regret curves
7. LLM confusion matrix
8. Accuracy vs latency scatter
9. Platform radar chart
10. End-to-end comparison
11. **Platform-experiment heatmap** (NEW)
12. **Agent learning curves** (NEW)
13. **Latency vs throughput scatter** (NEW)
14. **Platform efficiency** (NEW)
15. **Agent decision patterns** (NEW)
16. **Performance distributions** (NEW)
17. **Correlation heatmap** (NEW)
18. **Best platform scenarios** (NEW)

## Key Insights Provided

### Performance Analysis
- Which platforms are fastest for each operation
- Memory vs speed trade-offs
- CPU efficiency rankings

### Agent Intelligence
- Which agents make best decisions
- Agent learning and improvement over time
- Decision consistency and patterns

### Statistical Validity
- Significance tests between platforms/agents
- Confidence in performance differences
- Robust statistical comparisons

### Practical Recommendations
- Best platform for each use case
- Agent selection guidance
- Resource optimization strategies

## Usage

All new analysis is automatically generated when you run:
```bash
python app.py
```

Results are saved in:
```
experiments/runs/<timestamp>/
├── summary_*.csv (18 tables)
└── plots/
    └── *.png (18 visualizations)
```

## Dependencies

New dependency added:
- `scipy>=1.7.0` (for statistical tests)

Install with:
```bash
pip install -r requirements.txt
```

