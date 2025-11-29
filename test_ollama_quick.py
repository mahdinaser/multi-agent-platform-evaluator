#!/usr/bin/env python3
"""
Quick test script to verify Ollama LLM integration.
Runs minimal experiments to test LLM agent.
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

def main():
    """Run quick test with minimal experiments."""
    print("=" * 80)
    print("QUICK OLLAMA TEST - Minimal experiments to verify LLM integration")
    print("=" * 80)
    print()
    
    # Use test config
    config_path = project_root / 'config' / 'config_test.yaml'
    if not config_path.exists():
        print(f"ERROR: Test config not found: {config_path}")
        print("Please create config_test.yaml first")
        sys.exit(1)
    
    config = load_config(str(config_path))
    llm_config = config.get('llm_config', {})
    
    # Setup
    timestamp = get_timestamp()
    output_base = project_root / 'experiments' / 'runs' / f'test_{timestamp}'
    ensure_dir(output_base)
    
    log_file = output_base / 'logs.txt'
    setup_logging(str(log_file))
    logger = logging.getLogger(__name__)
    
    set_seed(config.get('experiment', {}).get('seed', 42))
    
    try:
        # Step 1: Generate minimal data
        print("Step 1: Generating minimal test data...")
        data_dir = project_root / 'data'
        ensure_dir(data_dir)
        generator = DataGenerator(str(data_dir))
        generator.generate_all(config, max_data_size=50000)
        print("✓ Data generated")
        print()
        
        # Step 2: Initialize runner
        print("Step 2: Initializing experiment runner...")
        runner = ExperimentRunner(str(data_dir), str(output_base), llm_config=llm_config)
        print("✓ Runner initialized")
        print()
        
        # Step 3: Run minimal experiments
        print("Step 3: Running minimal experiments (testing LLM agent)...")
        agents = config.get('agents', [])
        platforms = config.get('platforms', [])
        data_sources = []
        
        # Get actual data sources generated
        for source_type in ['tabular', 'logs', 'vectors', 'timeseries', 'text']:
            if source_type == 'tabular':
                for size in config.get('data', {}).get('tabular', {}).get('sizes', []):
                    data_sources.append(f'tabular_{size}')
            elif source_type in ['logs', 'vectors', 'timeseries', 'text']:
                data_sources.append(source_type)
        
        experiment_types = config.get('experiments', [])
        
        print(f"  Agents: {agents}")
        print(f"  Platforms: {platforms}")
        print(f"  Data sources: {data_sources}")
        print(f"  Experiment types: {experiment_types}")
        print()
        
        runner.run_all_experiments(agents, platforms, data_sources, experiment_types, config)
        print("✓ Experiments complete")
        print()
        
        # Step 4: Quick analysis
        print("Step 4: Generating quick analysis...")
        analysis = AnalysisEngine(str(output_base))
        analysis.generate_all_summaries()
        print("✓ Analysis complete")
        print()
        
        # Check if LLM was used
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        
        # Check logs for Ollama usage
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
                if '✓ Ollama initialized' in log_content or 'Ollama initialized' in log_content:
                    print("✅ SUCCESS: Ollama LLM was initialized and used!")
                elif 'POST http://127.0.0.1:11434/api/generate' in log_content:
                    print("✅ SUCCESS: Ollama API calls detected - LLM is working!")
                elif 'Found' in log_content and 'Ollama model' in log_content:
                    print("✅ SUCCESS: Ollama models detected!")
                else:
                    print("⚠️  WARNING: Could not confirm Ollama usage in logs")
        
        print(f"\nOutput directory: {output_base}")
        print(f"Log file: {log_file}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

