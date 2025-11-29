# Multi-Agent Platform Evaluator

Comprehensive evaluation framework for multi-agent AI systems in adaptive database platform selection, comparing 5 agent frameworks, 11 LLM models, and 7 platforms with tool-augmented decision-making and extensive benchmarking.

## Overview

This framework evaluates different AI agent strategies for selecting optimal data processing platforms (Pandas, DuckDB, Polars, SQLite, FAISS, Annoy) based on workload characteristics. It supports multiple agent frameworks, LLM models, and provides comprehensive benchmarking and statistical analysis.

## Key Features

- **5 Agent Frameworks**: MCP, LangChain, LangGraph, FastAgency, AutoGen
- **11 LLM Models**: Llama, Qwen, DeepSeek, Mistral, Phi, Gemma families (2B to 70B parameters)
- **7 Platforms**: Pandas, DuckDB, Polars, SQLite, FAISS, Annoy, Baseline
- **Tool-Augmented Decisions**: Performance history, platform comparison, specifications
- **Comprehensive Benchmarking**: TPC-H, NYC Taxi, synthetic workloads
- **Statistical Analysis**: Confidence intervals, significance testing, regret analysis
- **Multi-Model Testing**: Automatic agent generation for each LLM model
- **Stateful Pipelines**: LangGraph for complex multi-step decisions
- **Multi-Agent Collaboration**: AutoGen for collaborative decision-making

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mahdinaser/multi-agent-platform-evaluator.git
cd multi-agent-platform-evaluator

# Install dependencies
pip install -r requirements.txt

# Optional: Install agent frameworks
pip install langchain langchain-community langgraph pyautogen

# Optional: Install and setup Ollama for LLM agents
# See LLM_SETUP.md for details
```

### Running Experiments

```bash
# Run with default configuration
python app.py

# Or use quick mode for faster testing
# Edit config.yaml: set quick_mode: true
```

## Project Structure

```
multi-agent-platform-evaluator/
├── agents/              # Agent implementations
│   ├── agent_llm.py     # Base LLM agent
│   ├── agent_llm_mcp.py # MCP tool-augmented agent
│   ├── agent_langchain.py
│   ├── agent_langgraph.py
│   ├── agent_fastagency.py
│   ├── agent_autogen.py
│   └── ...              # Other agents
├── platforms/           # Platform backends
│   ├── pandas_backend.py
│   ├── duckdb_backend.py
│   ├── polars_backend.py
│   └── ...
├── src/                 # Core framework
│   ├── agent_manager.py
│   ├── platform_manager.py
│   ├── experiment_runner.py
│   ├── analysis.py
│   └── plotting.py
├── config/              # Configuration
│   └── config.yaml
├── experiments/         # Experiment results
└── report/             # Generated reports
```

## Configuration

Edit `config/config.yaml` to customize:

- Agent types and models
- Platforms to test
- Experiment types and data sizes
- LLM configuration (Ollama models)
- Multi-model testing

## Documentation

- `AGENT_FRAMEWORKS.md` - Agent framework comparison
- `MULTI_LLM_TESTING.md` - Multi-model LLM testing guide
- `COMPLETE_MODEL_INVENTORY.md` - All supported LLM models
- `MCP_INTEGRATION.md` - MCP tool integration details
- `LLM_SETUP.md` - LLM setup instructions

## Research Contributions

- First systematic comparison of agent frameworks for system optimization
- Novel evaluation of tool-augmented LLMs (MCP) for database selection
- Comprehensive analysis of 11 LLM models (2B to 70B parameters)
- Multi-agent vs single-agent comparison study
- Stateful vs stateless decision-making analysis

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies
- Optional: Ollama for local LLM support
- Optional: LangChain, LangGraph, AutoGen for framework agents

## License

[Add your license here]

## Citation

[Add citation information when paper is published]

## Contact

[Add contact information]
