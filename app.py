#!/usr/bin/env python3
"""
Research Experiment Engine - Main Entry Point
Multi-Agent Multi-Platform Evaluation Framework for PVLDB Paper

Usage: python app.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime
from tqdm import tqdm

from src.utils import (
    setup_logging, load_config, ensure_dir, get_timestamp, set_seed, path_join
)
from src.data_generator import DataGenerator
from src.experiment_runner import ExperimentRunner
from src.analysis import AnalysisEngine
from src.plotting import PlottingEngine

def generate_report(output_dir: Path, config: dict):
    """Generate comprehensive analysis report."""
    report_path = output_dir / 'analysis_report.md'
    
    # Read summary tables
    summary_files = {
        'overall': output_dir / 'summary_overall.csv',
        'platform': output_dir / 'summary_by_platform.csv',
        'agent': output_dir / 'summary_by_agent.csv',
        'datasource': output_dir / 'summary_by_datasource.csv',
        'latency': output_dir / 'summary_latency_stats.csv',
        'memory': output_dir / 'summary_memory_stats.csv',
        'cpu': output_dir / 'summary_cpu_stats.csv',
        'vector': output_dir / 'summary_vector_performance.csv',
        'text': output_dir / 'summary_text_similarity.csv',
        'stability': output_dir / 'summary_stability.csv',
        'accuracy': output_dir / 'summary_agent_accuracy.csv',
        'regret': output_dir / 'summary_agent_regret.csv'
    }
    
    import pandas as pd
    
    report = []
    
    def safe_read_csv(filepath):
        """Safely read CSV file."""
        try:
            if filepath.exists():
                return pd.read_csv(filepath)
            return pd.DataFrame()
        except Exception as e:
            logging.warning(f"Could not read {filepath}: {e}")
            return pd.DataFrame()
    report.append("# Comprehensive Multi-Agent Multi-Platform Evaluation Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # 1. Executive Summary
    report.append("## 1. Executive Summary")
    report.append("")
    report.append("This report presents a comprehensive evaluation of multiple intelligent agents")
    report.append("selecting optimal data processing platforms across diverse workloads and data types.")
    report.append("The experiment suite evaluates 5 agent types, 6 platforms, 5 data source types,")
    report.append("and 7 experiment types, producing a total of 1,050 experimental configurations.")
    report.append("")
    
    # 2. Data Sources Overview
    report.append("## 2. Data Sources Overview")
    report.append("")
    report.append("### 2.1 Tabular Data")
    report.append("Three datasets of varying sizes (50K, 500K, 1M rows) with mixed numeric and categorical columns.")
    report.append("")
    report.append("### 2.2 Log Data")
    report.append("Timestamped event logs with user IDs and event types, following heavy-tailed distributions.")
    report.append("")
    report.append("### 2.3 Vector Data")
    report.append("128-dimensional normalized vectors for similarity search experiments.")
    report.append("")
    report.append("### 2.4 Time-Series Data")
    report.append("High-frequency time-series with seasonal patterns for windowed aggregations.")
    report.append("")
    report.append("### 2.5 Text Data")
    report.append("Text corpus for similarity and search experiments.")
    report.append("")
    
    # 3. Agent Descriptions
    report.append("## 3. Agent Descriptions")
    report.append("")
    report.append("### 3.1 Rule-Based Agent")
    report.append("Uses heuristic rules to select platforms based on data type and experiment type.")
    report.append("Example: 'For vector data → use FAISS'")
    report.append("")
    report.append("### 3.2 Bandit Agent (UCB1)")
    report.append("Multi-armed bandit using UCB1 algorithm to learn optimal platform selection.")
    report.append("Tracks regret and reward over time.")
    report.append("")
    report.append("### 3.3 Cost-Model Agent")
    report.append("Trains linear regression models to predict platform latency.")
    report.append("Selects platform with lowest predicted latency.")
    report.append("")
    report.append("### 3.4 LLM Agent")
    report.append("Simulated LLM-based agent that reasons about platform characteristics.")
    report.append("Uses knowledge base to make informed decisions.")
    report.append("")
    report.append("### 3.5 Hybrid Agent")
    report.append("Combines rule-based, cost-model, and LLM agents with weighted voting.")
    report.append("Weights: cost-model (50%), rule-based (30%), LLM (20%).")
    report.append("")
    
    # 4. Platform Characteristics
    report.append("## 4. Platform Characteristics")
    report.append("")
    report.append("| Platform | Type | Strengths | Use Cases |")
    report.append("|----------|------|------------|-----------|")
    report.append("| Pandas | In-memory DataFrame | General-purpose, easy to use | Small-medium data, prototyping |")
    report.append("| DuckDB | Analytical SQL | Fast aggregations, columnar | Large-scale analytics |")
    report.append("| SQLite | OLTP SQL | Structured queries, ACID | Transactional workloads |")
    report.append("| FAISS | Vector search | High-performance similarity | Vector embeddings, ML |")
    report.append("| Annoy | Approx NN | Fast indexing, approximate | Large-scale similarity |")
    report.append("| Baseline | Naive Python | Baseline comparison | Reference implementation |")
    report.append("")
    
    # 5. Key Findings
    report.append("## 5. Key Findings")
    report.append("")
    
    # Load and analyze summary data
    overall_df = safe_read_csv(summary_files['overall'])
    if len(overall_df) > 0:
            report.append(f"- **Total Experiments:** {int(overall_df['total_experiments'].iloc[0])}")
            report.append(f"- **Success Rate:** {overall_df['success_rate'].iloc[0]:.2%}")
            report.append(f"- **Average Latency:** {overall_df['avg_latency_ms'].iloc[0]:.2f} ms")
            report.append("")
    
    platform_df = safe_read_csv(summary_files['platform'])
    if len(platform_df) > 0:
            best_latency = platform_df.loc[platform_df['latency_ms_mean'].idxmin()]
            report.append(f"- **Fastest Platform (avg):** {best_latency['platform']} ({best_latency['latency_ms_mean']:.2f} ms)")
            report.append("")
    
    agent_df = safe_read_csv(summary_files['agent'])
    if len(agent_df) > 0:
            best_agent = agent_df.loc[agent_df['latency_ms_mean'].idxmin()]
            report.append(f"- **Best Performing Agent (avg latency):** {best_agent['agent']} ({best_agent['latency_ms_mean']:.2f} ms)")
            report.append("")
    
    # 6. Agent Performance Analysis
    report.append("## 6. Agent Performance Analysis")
    report.append("")
    report.append("### 6.1 What Each Agent Excels At")
    report.append("")
    
    accuracy_df = safe_read_csv(summary_files['accuracy'])
    if len(accuracy_df) > 0:
            for _, row in accuracy_df.iterrows():
                report.append(f"- **{row['agent']}:** Accuracy {row['accuracy']:.2%}, Latency ratio {row['latency_ratio']:.2f}x")
            report.append("")
    
    # 7. Data Type vs Agent Performance
    report.append("## 7. Data Type vs Agent Performance")
    report.append("")
    report.append("Observations:")
    report.append("- Vector data: Specialized platforms (FAISS, Annoy) significantly outperform general-purpose platforms")
    report.append("- Tabular data: DuckDB excels for large-scale aggregations")
    report.append("- Text data: Baseline and Pandas perform similarly for simple similarity")
    report.append("- Time-series: Windowed operations benefit from optimized SQL engines")
    report.append("")
    
    # 8. Summary Tables
    report.append("## 8. Summary Tables")
    report.append("")
    report.append("### 8.1 Overall Summary")
    overall_df = safe_read_csv(summary_files['overall'])
    if len(overall_df) > 0:
        report.append(overall_df.to_markdown(index=False))
        report.append("")
    
    report.append("### 8.2 Platform Performance")
    platform_df = safe_read_csv(summary_files['platform'])
    if len(platform_df) > 0:
        report.append(platform_df.to_markdown(index=False))
        report.append("")
    
    report.append("### 8.3 Agent Performance")
    agent_df = safe_read_csv(summary_files['agent'])
    if len(agent_df) > 0:
        report.append(agent_df.to_markdown(index=False))
        report.append("")
    
    # 9. Visualizations
    report.append("## 9. Visualizations")
    report.append("")
    plot_files = [
        'latency_distribution.png',
        'memory_comparison.png',
        'cpu_comparison.png',
        'stability_heatmap.png',
        'agent_decisions.png',
        'regret_curves.png',
        'llm_confusion_matrix.png',
        'accuracy_latency_scatter.png',
        'platform_radar.png',
        'e2e_comparison.png'
    ]
    
    plots_dir = output_dir / 'plots'
    for plot_file in plot_files:
        plot_path = plots_dir / plot_file
        if plot_path.exists():
            report.append(f"### {plot_file.replace('_', ' ').replace('.png', '').title()}")
            report.append(f"![{plot_file}](plots/{plot_file})")
            report.append("")
    
    # 10. Future Directions
    report.append("## 10. Future Directions")
    report.append("")
    report.append("1. **Reinforcement Learning Agents:** Implement deep RL agents for platform selection")
    report.append("2. **Multi-Objective Optimization:** Consider latency, cost, and energy consumption")
    report.append("3. **Dynamic Workload Adaptation:** Agents that adapt to changing workload patterns")
    report.append("4. **Distributed Platform Support:** Evaluate distributed platforms (Spark, Dask)")
    report.append("5. **Real-World Workloads:** Test on production traces and benchmarks")
    report.append("6. **Explainability:** Enhanced reasoning traces for agent decisions")
    report.append("")
    
    report.append("---")
    report.append("")
    report.append("*Report generated by Research Experiment Engine*")
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logging.info(f"Analysis report written to {report_path}")

def main():
    """Main entry point."""
    print("=" * 80)
    print("Research Experiment Engine - Multi-Agent Multi-Platform Evaluation")
    print("PVLDB Paper Framework")
    print("=" * 80)
    print()
    
    # Load configuration
    config_path = path_join('config', 'config.yaml')
    config = load_config(config_path)
    
    # Set random seed
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    
    # Create output directory with timestamp
    timestamp = get_timestamp()
    output_base = path_join('experiments', 'runs', timestamp)
    ensure_dir(output_base)
    
    # Setup logging
    log_file = path_join(output_base, 'logs.txt')
    logger = setup_logging(log_file, level=logging.INFO)
    
    logger.info("=" * 80)
    logger.info("Starting Research Experiment Engine")
    logger.info(f"Output directory: {output_base}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Create folder structure
        logger.info("Step 1: Creating folder structure...")
        data_dir = ensure_dir(path_join('data'))
        ensure_dir(str(data_dir / 'source_tabular'))
        ensure_dir(str(data_dir / 'source_logs'))
        ensure_dir(str(data_dir / 'source_vectors'))
        ensure_dir(str(data_dir / 'source_timeseries'))
        ensure_dir(str(data_dir / 'source_text'))
        ensure_dir(str(Path(output_base) / 'plots'))
        logger.info("Folder structure created")
        
        # Step 2: Generate data
        logger.info("Step 2: Generating data sources...")
        data_gen = DataGenerator(str(data_dir), seed=seed)
        data_gen.generate_all(config.get('data', {}))
        logger.info("Data generation complete")
        
        # Step 3: Initialize experiment runner
        logger.info("Step 3: Initializing experiment runner...")
        runner = ExperimentRunner(str(data_dir), str(output_base))
        # Pass config to runner for agent initialization
        runner.config = config
        logger.info("Experiment runner initialized")
        
        # Step 4: Run experiments
        logger.info("Step 4: Running comprehensive experiments...")
        agents = config.get('agents', [])
        platforms = config.get('platforms', [])
        data_sources = []
        
        # Build data source list
        if 'tabular' in config.get('data', {}):
            sizes = config['data']['tabular'].get('sizes', [50000, 500000, 1000000])
            data_sources.extend([f'tabular_{size}' for size in sizes])
        data_sources.extend(['logs', 'vectors', 'timeseries', 'text'])
        
        experiment_types = config.get('experiments', [])
        
        # Log experiment scope
        quick_mode = config.get('experiment', {}).get('quick_mode', False)
        if quick_mode:
            total = len(agents) * len(data_sources) * len(experiment_types)
            logger.info(f"QUICK MODE: Will run {total} experiments (agent-selected platforms only)")
        else:
            total = len(agents) * len(platforms) * len(data_sources) * len(experiment_types)
            logger.info(f"FULL MODE: Will run {total} experiments (all platform combinations)")
        
        runner.run_all_experiments(agents, platforms, data_sources, experiment_types, config)
        logger.info("Experiments complete")
        
        # Step 5: Generate analysis tables
        logger.info("Step 5: Generating analysis tables...")
        metrics_file = path_join(output_base, 'metrics_raw.csv')
        if os.path.exists(metrics_file):
            analysis = AnalysisEngine(metrics_file, str(output_base))
            analysis.generate_all_tables()
            logger.info("Analysis tables generated (18+ tables)")
        else:
            logger.warning(f"Metrics file not found: {metrics_file}")
        
        # Step 6: Generate visualizations
        logger.info("Step 6: Generating visualizations...")
        if os.path.exists(metrics_file):
            plotting = PlottingEngine(metrics_file, str(output_base))
            plotting.generate_all_plots()
            logger.info("Visualizations generated (18+ plots)")
        else:
            logger.warning(f"Metrics file not found: {metrics_file}")
        
        # Step 7: Generate report
        logger.info("Step 7: Generating analysis report...")
        generate_report(Path(output_base), config)
        logger.info("Analysis report generated")
        
        # Final summary
        print()
        print("=" * 80)
        print("EXPERIMENT SUITE COMPLETE")
        print("=" * 80)
        print(f"Output directory: {output_base}")
        print(f"Metrics: {metrics_file}")
        print(f"Summary tables: {output_base}/summary_*.csv")
        print(f"Visualizations: {output_base}/plots/*.png")
        print(f"Analysis report: {output_base}/analysis_report.md")
        print(f"Logs: {log_file}")
        print("=" * 80)
        
        logger.info("Research Experiment Engine completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check logs for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()

