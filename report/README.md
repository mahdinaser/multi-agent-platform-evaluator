# IEEE Conference Paper: Multi-Agent Platform Selection

This directory contains a comprehensive IEEE format research paper based on the multi-agent platform evaluation experiment results.

## Paper Details

**Title:** Multi-Agent Platform Selection for Data Processing: A Comparative Study of Intelligent Decision-Making Strategies

**Format:** IEEE Conference Paper (IEEEtran class)

**Length:** ~10 pages

**Sections:**
- Abstract
- Introduction
- Related Work
- Methodology
- Experimental Setup
- Results
- Discussion
- Conclusion
- References (23 citations)

## Compilation

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- IEEEtran document class

### Compile the paper

**Option 1: Using Make**
```bash
make
```

**Option 2: Manual compilation**
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

**Option 3: Using latexmk**
```bash
latexmk -pdf paper.tex
```

### Clean auxiliary files
```bash
make clean
```

## Paper Contents

### Key Contributions
1. Comprehensive evaluation of 5 intelligent agent strategies across 245 experimental configurations
2. First integration of LLM (Ollama llama2) for real-time platform selection
3. Statistical validation showing context-dependent platform performance
4. Open-source framework with 18+ analytical metrics

### Experimental Results
- **Cost-Model Agent:** 20.55 ms average latency (best)
- **LLM Agent:** 21.20 ms average latency (2nd best, 100% accuracy)
- **Hybrid Agent:** 23.49 ms (effective ensemble)
- **Rule-Based Agent:** 25.84 ms
- **Bandit Agent:** 904.95 ms (high variance due to exploration)

### Statistical Analysis
- 245 total experiments
- 140 successful experiments (57.14% success rate)
- No significant platform differences (p > 0.05)
- Logarithmic regret convergence for bandit agent

## Data Sources

All data is derived from the experiment run:
- `experiments/runs/20251127_232031/`
- Summary tables (18+ CSV files)
- Performance metrics
- Agent decisions
- Platform comparisons

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{multiagent2024,
  title={Multi-Agent Platform Selection for Data Processing: A Comparative Study of Intelligent Decision-Making Strategies},
  author={Anonymous},
  booktitle={Proceedings of IEEE Conference},
  year={2024}
}
```

## License

This work is part of the PVLDB Multi-Agent Platform Evaluator research project.

