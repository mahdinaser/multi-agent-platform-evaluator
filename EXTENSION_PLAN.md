# Multi-Agent Platform Selection - Extension Plan

## Overview
Comprehensive extension to add new platforms, agents, benchmarks, and experimental rigor.

## Phase 1: High Impact, Easy Changes ✅

### 1.1 Multiple Runs with Confidence Intervals (PRIORITY 1)
**Impact:** High | **Effort:** Easy | **Status:** In Progress

- [x] Add `num_runs: 10` and `warmup_runs: 3` to config
- [ ] Update experiment_runner.py to run each config multiple times
- [ ] Calculate mean, std, 95% confidence intervals
- [ ] Update metrics to include CI columns
- [ ] Update analysis to show error bars

**Files to modify:**
- `config/config.yaml`
- `src/experiment_runner.py`
- `src/analysis.py`
- `src/plotting.py`

### 1.2 Separate Decision Time Tracking (PRIORITY 9)
**Impact:** Medium | **Effort:** Easy

- [ ] Add `decision_time_ms` metric
- [ ] Track agent decision time separately from query execution
- [ ] Update metrics collection
- [ ] Add decision overhead analysis

**Files to modify:**
- `src/metrics.py`
- `src/experiment_runner.py`
- `src/analysis.py`

## Phase 2: Critical Platform Additions ✅

### 2.1 Fix Existing Platforms (PRIORITY 2)
**Impact:** Critical | **Effort:** Medium

**DuckDB Backend:**
- [ ] Fix class name: `DuckdbBackend` → `DuckDBBackend`
- [ ] Implement all query types
- [ ] Add connection pooling

**SQLite Backend:**
- [ ] Fix class name: `SqliteBackend` → `SQLiteBackend`  
- [ ] Implement all query types
- [ ] Add indexing support

**FAISS Backend:**
- [ ] Fix class name: `FaissBackend` → `FAISSBackend`
- [ ] Implement vector operations
- [ ] Add index type selection

**Files to modify:**
- `platforms/duckdb_backend.py`
- `platforms/sqlite_backend.py`
- `platforms/faiss_backend.py`
- `src/platform_manager.py`

### 2.2 Add Polars Platform (PRIORITY 2)
**Impact:** High | **Effort:** Easy

- [ ] Create `platforms/polars_backend.py`
- [ ] Implement PolarsBackend class
- [ ] Add to platform_manager
- [ ] Test performance vs Pandas

## Phase 3: Baseline Agents ✅

### 3.1 Add Simple Baseline Agents (PRIORITY 3)
**Impact:** Critical | **Effort:** Easy

**Random Agent:**
- [ ] Create `agents/agent_random.py`
- [ ] Random platform selection
- [ ] Use for performance floor

**Oracle Agent:**
- [ ] Create `agents/agent_oracle.py`
- [ ] Always selects optimal (requires post-hoc knowledge)
- [ ] Use for performance ceiling

**Static-Best Agent:**
- [ ] Create `agents/agent_static_best.py`
- [ ] Selects historically best platform
- [ ] Train on initial warm-up phase

**Round-Robin Agent:**
- [ ] Create `agents/agent_round_robin.py`
- [ ] Cycles through platforms
- [ ] Fair baseline

### 3.2 Add Advanced Agents (PRIORITY 6)
**Impact:** High | **Effort:** Medium

**LinUCB (Contextual Bandit):**
- [ ] Create `agents/agent_linucb.py`
- [ ] Feature extraction (data size, query type, etc.)
- [ ] Contextual upper confidence bound
- [ ] Ridge regression for reward prediction

**Thompson Sampling:**
- [ ] Create `agents/agent_thompson.py`
- [ ] Beta distribution per platform
- [ ] Bayesian exploration
- [ ] Compare with UCB1

**Files to create:**
- `agents/agent_random.py`
- `agents/agent_oracle.py`
- `agents/agent_static_best.py`
- `agents/agent_round_robin.py`
- `agents/agent_linucb.py`
- `agents/agent_thompson.py`

## Phase 4: Benchmark Data ✅

### 4.1 TPC-H Benchmark (PRIORITY 4)
**Impact:** Critical | **Effort:** Medium

- [ ] Add TPC-H data generator
- [ ] Support SF1 (1GB) and SF10 (10GB) scale factors
- [ ] Generate 8 tables (lineitem, orders, customer, part, supplier, partsupp, nation, region)
- [ ] Implement 22 TPC-H queries
- [ ] Add to data_generator.py or create separate tpch_generator.py

**Queries to implement:**
- Q1: Pricing Summary Report
- Q3: Shipping Priority
- Q6: Forecasting Revenue Change
- Q12: Shipping Modes and Order Priority
- (Select subset of representative queries)

### 4.2 NYC Taxi Benchmark (PRIORITY 6)
**Impact:** High | **Effort:** Medium

- [ ] Download NYC Taxi dataset (or generate synthetic)
- [ ] Load parquet files
- [ ] Typical queries: aggregations, time-window, spatial
- [ ] Add to benchmark suite

**Files to create:**
- `src/benchmarks/tpch_generator.py`
- `src/benchmarks/tpch_queries.py`
- `src/benchmarks/nyc_taxi_loader.py`
- `src/benchmarks/nyc_taxi_queries.py`

## Phase 5: Scalability Tests ✅

### 5.1 Data Size Scaling (PRIORITY 5)
**Impact:** High | **Effort:** Medium

- [ ] Add scalability experiment mode
- [ ] Test sizes: 10K, 50K, 100K, 500K, 1M, 5M, 10M rows
- [ ] Same query across all sizes
- [ ] Plot latency vs data size
- [ ] Identify platform crossover points

**Files to modify:**
- `config/config.yaml` - add scalability config
- `src/experiment_runner.py` - add scalability mode
- `src/analysis.py` - add scalability analysis
- `src/plotting.py` - add scaling plots

## Phase 6: Enhanced Analysis ✅

### 6.1 Confidence Intervals (PRIORITY 1)
- [ ] Calculate 95% CI for all metrics
- [ ] Add error bars to all plots
- [ ] Statistical significance tests (t-tests, ANOVA)
- [ ] Effect size calculations (Cohen's d)

### 6.2 Agent Decision Analysis
- [ ] Decision time vs accuracy trade-off
- [ ] Exploration rate over time
- [ ] Regret bounds verification
- [ ] Feature importance (for contextual bandits)

### 6.3 Platform Performance Profiling
- [ ] Per-query breakdown
- [ ] Scalability curves
- [ ] Memory/CPU profiling
- [ ] Pareto frontier (latency vs resource)

## Implementation Order (By Priority)

1. ✅ **Multiple runs + CI** (Phase 1.1) - Start here
2. ✅ **Fix DuckDB, SQLite, FAISS** (Phase 2.1) - Critical
3. ✅ **Add Polars** (Phase 2.2) - Easy win
4. ✅ **Add baseline agents** (Phase 3.1) - Critical baselines
5. ✅ **TPC-H benchmark** (Phase 4.1) - Real workload
6. ✅ **Decision time tracking** (Phase 1.2) - Easy, useful
7. ✅ **Scalability tests** (Phase 5.1) - Important analysis
8. ✅ **LinUCB + Thompson** (Phase 3.2) - Advanced agents
9. ✅ **NYC Taxi** (Phase 4.2) - Additional benchmark
10. ✅ **Enhanced analysis** (Phase 6) - Final polish

## Estimated Timeline

- **Phase 1** (Multiple runs + decision time): 2-3 hours
- **Phase 2** (Fix platforms + Polars): 3-4 hours
- **Phase 3.1** (Baseline agents): 2-3 hours
- **Phase 3.2** (Advanced agents): 4-5 hours
- **Phase 4.1** (TPC-H): 5-6 hours
- **Phase 4.2** (NYC Taxi): 3-4 hours
- **Phase 5** (Scalability): 2-3 hours
- **Phase 6** (Enhanced analysis): 3-4 hours

**Total estimated time:** 24-32 hours

## Expected Results

### New Experimental Scale
- **Platforms:** 3 → 7 (Pandas, Annoy, Baseline, DuckDB, Polars, SQLite, FAISS)
- **Agents:** 5 → 11 (existing + Random, Oracle, Static-Best, Round-Robin, LinUCB, Thompson)
- **Data Sources:** 7 → 15+ (existing + TPC-H SF1/SF10, NYC Taxi, scalability tests)
- **Runs per config:** 1 → 10 (with 3 warm-ups)
- **Total experiments:** ~245 → ~10,000+

### New Metrics
- 95% confidence intervals for all metrics
- Decision time (ms)
- Exploration rate
- Regret bounds
- Scalability coefficients

### New Plots
- Error bars on all comparisons
- Scalability curves (log-log)
- Decision overhead analysis
- Pareto frontiers
- Feature importance (LinUCB)
- Exploration vs exploitation over time

## Notes

- Start with Phase 1 (multiple runs) for immediate impact
- Platform fixes (Phase 2) are critical - many experiments currently failing
- TPC-H is industry-standard and critical for credibility
- LinUCB adds significant research value
- All changes maintain backward compatibility

## Success Criteria

- [x] All 7 platforms working
- [ ] All 11 agents implemented
- [ ] 10 runs per config with < 5% CI width
- [ ] TPC-H queries working
- [ ] Scalability tests from 10K to 10M
- [ ] Decision time < 5% of total time (except LLM)
- [ ] Paper updated with new results

