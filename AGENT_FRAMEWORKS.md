# Agent Framework Integration

## Overview

The framework now supports **5 different agent frameworks** for LLM-based platform selection, each with unique strengths:

1. **MCP (Model Context Protocol)** - Original tool-augmented LLM
2. **LangChain** - Most flexible framework
3. **LangGraph** - Stateful, reliable agentic pipelines
4. **FastAgency** - Lightweight, tool-focused, ideal for local dev
5. **AutoGen** - Best for multi-agent local conversations + tools

## Framework Comparison

| Framework | Strengths | Best For | Complexity |
|-----------|-----------|----------|------------|
| **MCP** | Simple, direct tool access | Quick tool integration | Low |
| **LangChain** | Most flexible, extensive ecosystem | Complex workflows, many tools | Medium |
| **LangGraph** | Stateful pipelines, reliable | Multi-step decisions, state management | Medium-High |
| **FastAgency** | Lightweight, fast, simple | Local development, quick prototyping | Low |
| **AutoGen** | Multi-agent collaboration | Collaborative decision-making | High |

## Framework Details

### 1. MCP (Model Context Protocol)

**File:** `agents/agent_llm_mcp.py`

**Features:**
- Direct tool access (performance history, platform comparison, specs)
- Simple prompt enhancement with tool results
- Lightweight integration

**Usage:**
```yaml
agents:
  - llm_mcp
```

**Tools:**
- `get_performance_history` - Historical performance metrics
- `compare_platforms` - Side-by-side platform comparison
- `get_platform_specs` - Platform characteristics

---

### 2. LangChain

**File:** `agents/agent_langchain.py`

**Features:**
- Most flexible framework
- Extensive tool ecosystem
- Zero-shot ReAct agent pattern
- Automatic tool selection

**Usage:**
```yaml
agents:
  - langchain
```

**Tools:**
- `get_performance_history` - Historical performance
- `compare_platforms` - Platform comparison
- `get_platform_specs` - Platform specifications

**Installation:**
```bash
pip install langchain langchain-community
```

**Architecture:**
- Uses `AgentType.ZERO_SHOT_REACT_DESCRIPTION`
- LLM automatically decides which tools to use
- Supports reasoning and acting in loops

---

### 3. LangGraph

**File:** `agents/agent_langgraph.py`

**Features:**
- Stateful agentic pipelines
- Reliable state management
- Multi-step decision workflows
- Graph-based execution

**Usage:**
```yaml
agents:
  - langgraph
```

**Pipeline:**
1. **Gather Data** - Collect historical performance
2. **Analyze** - Process data and select platform
3. **End** - Return decision

**Installation:**
```bash
pip install langgraph langchain-community
```

**Architecture:**
- StateGraph with TypedDict state
- Nodes: `gather_data` → `analyze` → `END`
- State persists across steps

---

### 4. FastAgency

**File:** `agents/agent_fastagency.py`

**Features:**
- Lightweight and fast
- Tool-focused design
- Ideal for local development
- Minimal dependencies

**Usage:**
```yaml
agents:
  - fastagency
```

**Approach:**
- Direct tool function calls (no heavy framework)
- Simple prompt building with tool results
- Fast execution

**Architecture:**
- Lightweight tool execution
- Direct Ollama integration
- No heavy framework overhead

---

### 5. AutoGen

**File:** `agents/agent_autogen.py`

**Features:**
- Multi-agent collaboration
- Specialized agents for different roles
- Group chat for decision-making
- Best for complex collaborative tasks

**Usage:**
```yaml
agents:
  - autogen
```

**Agents:**
1. **Performance Analyst** - Analyzes historical data
2. **Platform Specialist** - Knows platform specs
3. **Decision Maker** - Synthesizes and decides

**Installation:**
```bash
pip install pyautogen
```

**Architecture:**
- GroupChat with 3 specialized agents
- Agents collaborate via conversation
- Decision Maker makes final choice

---

## Configuration

### Enable All Frameworks

```yaml
agents:
  - llm  # Baseline LLM
  - llm_mcp  # MCP tools
  - langchain  # LangChain framework
  - langgraph  # LangGraph pipelines
  - fastagency  # FastAgency lightweight
  - autogen  # AutoGen multi-agent
```

### Multi-Model Support

All frameworks support multi-model testing:

```yaml
llm_config:
  enable_multi_model: true
  test_models:
    - "llama2:latest"
    - "qwen3:14b"
    # ... more models
```

This creates agents like:
- `langchain_llama2_latest`
- `langgraph_qwen3_14b`
- `fastagency_deepseek_r1_14b`
- etc.

---

## Research Questions

### Q1: Which framework is best for system optimization?
**Compare:** All 5 frameworks on same tasks
- Measure: Mean latency of selected platforms
- Hypothesis: More sophisticated frameworks (LangGraph, AutoGen) outperform simpler ones

### Q2: Does framework complexity matter?
**Compare:** Simple (MCP, FastAgency) vs Complex (LangGraph, AutoGen)
- Track: Decision quality vs decision time
- Find: Sweet spot for production deployment

### Q3: Multi-agent vs single-agent?
**Compare:** AutoGen (multi-agent) vs others (single-agent)
- Hypothesis: Multi-agent collaboration improves decisions
- Test: Do specialized agents help?

### Q4: Stateful vs stateless?
**Compare:** LangGraph (stateful) vs others (stateless)
- With complex decisions, does state management help?
- Test: Multi-step platform selection

### Q5: Tool integration patterns?
**Compare:** How each framework uses tools
- MCP: Direct tool calls
- LangChain: Automatic tool selection
- LangGraph: Tools in stateful pipeline
- FastAgency: Lightweight tool calls
- AutoGen: Tools via agent conversations

---

## Expected Results

### Performance Hierarchy (Predicted)

```
Platform Selection Quality
     ↑
     │  AutoGen (multi-agent collaboration)
     │  ═══════════════════
     │  LangGraph (stateful pipelines)
     │  ───────────────────
     │  LangChain (flexible tools)
     │  ───────────────────
     │  MCP (direct tools)
     │  FastAgency (lightweight)
     │  ═══════════════════
     └──────────────────────→ Decision Time
                              Faster
```

### Decision Time (Predicted)

```
Decision Latency (ms)
     ↑
     │  AutoGen    ← 2000-5000ms (multi-agent)
     │  ════════════
     │  LangGraph  ← 1000-2000ms (stateful)
     │  ───────────
     │  LangChain  ← 500-1000ms (flexible)
     │  ───────────
     │  MCP        ← 300-800ms (direct)
     │  FastAgency ← 200-500ms (lightweight)
     │  ════════════
     └──────────────────→ Time
```

---

## Analysis Tables

The framework will generate:

### 1. **Framework Comparison Table**
| Framework | Mean Latency | Decision Time | Accuracy | Regret |
|-----------|-------------|---------------|----------|--------|
| AutoGen | ? | ? | ? | ? |
| LangGraph | ? | ? | ? | ? |
| LangChain | ? | ? | ? | ? |
| MCP | ? | ? | ? | ? |
| FastAgency | ? | ? | ? | ? |

### 2. **Framework Characteristics**
| Framework | Type | Complexity | Speed | Quality | Best Use Case |
|-----------|------|------------|-------|---------|---------------|
| AutoGen | Multi-agent | High | Slow | Excellent | Complex decisions |
| LangGraph | Stateful | Medium-High | Medium | Very Good | Multi-step workflows |
| LangChain | Flexible | Medium | Medium | Very Good | General purpose |
| MCP | Direct tools | Low | Fast | Good | Quick integration |
| FastAgency | Lightweight | Low | Fastest | Good | Local dev |

---

## Plots

### 1. **Framework Comparison (Bar Chart)**
- X-axis: Frameworks
- Y-axis: Mean latency of selected platforms
- Grouped bars: With/without historical data

### 2. **Decision Time vs Quality (Scatter)**
- X-axis: Decision time (ms)
- Y-axis: Platform selection quality
- Points: Each framework
- Pareto frontier: Best quality/time trade-offs

### 3. **Framework Complexity vs Performance**
- X-axis: Framework complexity (1-5)
- Y-axis: Decision quality
- Points: Each framework
- Trend: Does complexity help?

---

## Installation

### All Frameworks
```bash
pip install langchain langchain-community langgraph pyautogen
```

### Individual Frameworks
```bash
# LangChain only
pip install langchain langchain-community

# LangGraph only
pip install langgraph langchain-community

# AutoGen only
pip install pyautogen
```

**Note:** FastAgency and MCP don't require additional packages (use Ollama directly).

---

## Status

- ✅ MCP agent (original)
- ✅ LangChain agent
- ✅ LangGraph agent
- ✅ FastAgency agent
- ✅ AutoGen agent
- ✅ Multi-model support for all frameworks
- ✅ Configuration updated
- ✅ Requirements updated

## Quick Start

```bash
# 1. Install frameworks (optional - agents fallback if not installed)
pip install langchain langchain-community langgraph pyautogen

# 2. Update config.yaml
# Add frameworks to agents list

# 3. Run experiments
python app.py
```

---

## Paper Contribution

This multi-framework comparison adds a valuable research dimension:

### Section: Agent Framework Comparison
**Novel insights:**
- First systematic comparison of agent frameworks for system optimization
- Quantifies framework complexity vs quality trade-off
- Shows multi-agent collaboration effectiveness
- Identifies which frameworks are best for database selection

### Key Findings (Expected)
1. **Multi-agent (AutoGen) is best but slowest** (collaboration helps)
2. **Stateful (LangGraph) helps for complex decisions** (state management matters)
3. **Lightweight (FastAgency) is fastest** (good for real-time)
4. **Framework choice depends on use case** (no one-size-fits-all)
5. **Tool integration patterns matter** (how tools are used affects quality)

---

## References

- LangChain: https://github.com/langchain-ai/langchain
- LangGraph: https://github.com/langchain-ai/langgraph
- FastAgency: https://github.com/jxnl/fastagency
- AutoGen: https://github.com/microsoft/autogen
- MCP: https://modelcontextprotocol.io/

