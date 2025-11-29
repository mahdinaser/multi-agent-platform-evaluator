"""
Experiment runner for executing comprehensive experiments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from tqdm import tqdm

from src.metrics import MetricsCollector, ExperimentMetrics
from src.platform_manager import PlatformManager
from src.agent_manager import AgentManager

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs comprehensive experiments."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.platform_manager = PlatformManager()
        self.agent_manager = None  # Will be initialized with config
        self.metrics_collector = MetricsCollector()
        self.config = None
        
        self.all_metrics = []
        self.decisions_trace = []
        self._data_cache = {}  # Cache loaded data to avoid reloading
    
    def load_data(self, data_source: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load data based on source type. Uses cache to avoid reloading."""
        # Check cache first
        if use_cache and data_source in self._data_cache:
            return self._data_cache[data_source]
        
        data = {}
        
        if data_source.startswith('tabular'):
            # Extract size from name like "tabular_50000"
            size = int(data_source.split('_')[1]) if '_' in data_source else 50000
            file_path = self.data_dir / 'source_tabular' / f'tabular_{size}.parquet'
            if file_path.exists():
                data['df'] = pd.read_parquet(file_path)
                data['type'] = 'tabular'
                data['size'] = len(data['df'])
                data['num_columns'] = len(data['df'].columns)
        
        elif data_source == 'logs':
            file_path = self.data_dir / 'source_logs' / 'logs.parquet'
            if file_path.exists():
                data['df'] = pd.read_parquet(file_path)
                data['type'] = 'logs'
                data['size'] = len(data['df'])
        
        elif data_source == 'vectors':
            file_path = self.data_dir / 'source_vectors' / 'vectors.npy'
            if file_path.exists():
                data['vectors'] = np.load(file_path)
                data['type'] = 'vectors'
                data['size'] = len(data['vectors'])
        
        elif data_source == 'timeseries':
            file_path = self.data_dir / 'source_timeseries' / 'timeseries.parquet'
            if file_path.exists():
                data['df'] = pd.read_parquet(file_path)
                data['type'] = 'timeseries'
                data['size'] = len(data['df'])
        
        elif data_source == 'text':
            file_path = self.data_dir / 'source_text' / 'text_corpus.parquet'
            if file_path.exists():
                data['df'] = pd.read_parquet(file_path)
                data['type'] = 'text'
                data['size'] = len(data['df'])
                data['texts'] = data['df']['text'].tolist()
        
        # Cache the data
        if use_cache:
            self._data_cache[data_source] = data
        
        return data
    
    def run_experiment(self, agent_name: str, platform_name: str, 
                      data_source: str, experiment_type: str,
                      config: Dict[str, Any] = None) -> ExperimentMetrics:
        """Run a single experiment."""
        if config is None:
            config = {}
        
        experiment_id = f"{agent_name}_{platform_name}_{data_source}_{experiment_type}"
        
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            agent=agent_name,
            platform=platform_name,
            data_source=data_source,
            experiment_type=experiment_type,
            config=config
        )
        
        try:
            # Load data (use cache to avoid reloading)
            data = self.load_data(data_source, use_cache=True)
            if not data:
                metrics.is_correct = False
                metrics.error_message = "Failed to load data"
                return metrics
            
            # Get platform and agent
            platform = self.platform_manager.get_platform(platform_name)
            agent = self.agent_manager.get_agent(agent_name)
            
            if not platform or not agent:
                metrics.is_correct = False
                metrics.error_message = "Platform or agent not available"
                return metrics
            
            # Prepare context for agent
            context = {
                'size': data.get('size', 0),
                'num_columns': data.get('num_columns', 0),
                'type': data.get('type', '')
            }
            
            # Start measurement
            self.metrics_collector.start_measurement()
            
            # Run experiment based on type
            result = None
            num_records = data.get('size', 0)
            
            if experiment_type == 'scan':
                if 'df' in data:
                    result = platform.run_scan(data['df'], config)
                    num_records = len(result)
            
            elif experiment_type == 'filter':
                if 'df' in data:
                    filter_config = config.get('filter', {
                        'predicate': {
                            'column': data['df'].columns[0],
                            'operator': '>',
                            'value': data['df'][data['df'].columns[0]].median()
                        }
                    })
                    result = platform.run_filter(data['df'], filter_config)
                    num_records = len(result)
            
            elif experiment_type == 'aggregate':
                if 'df' in data:
                    df = data['df']
                    # Find suitable columns for aggregation
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    # Use first categorical column for group_by, or first column if no categorical
                    group_by_col = categorical_cols[0] if len(categorical_cols) > 0 else (df.columns[0] if len(df.columns) > 0 else None)
                    
                    # Use first numeric column for aggregation, or count if no numeric columns
                    if len(numeric_cols) > 0:
                        agg_col = numeric_cols[0]
                        agg_func = 'mean'
                    else:
                        agg_col = group_by_col if group_by_col else df.columns[0] if len(df.columns) > 0 else None
                        agg_func = 'count'
                    
                    if group_by_col and agg_col:
                        agg_config = config.get('aggregate', {
                            'group_by': [group_by_col],
                            'aggregations': {
                                agg_col: agg_func
                            }
                        })
                        result = platform.run_aggregate(df, agg_config)
                        num_records = len(result) if result is not None else 0
                    else:
                        result = pd.DataFrame()
                        num_records = 0
            
            elif experiment_type == 'join':
                if 'df' in data:
                    # Create second dataframe for join
                    data2 = data['df'].sample(min(10000, len(data['df']))).copy()
                    
                    # Find a suitable join key (prefer 'id' or 'doc_id', otherwise use first column)
                    join_key = None
                    if 'id' in data['df'].columns:
                        join_key = 'id'
                    elif 'doc_id' in data['df'].columns:
                        join_key = 'doc_id'
                    elif len(data['df'].columns) > 0:
                        join_key = data['df'].columns[0]
                    
                    if join_key:
                        # Rename columns in data2 except the join key
                        data2 = data2.rename(columns={col: f'{col}_2' for col in data2.columns if col != join_key})
                        join_config = config.get('join', {
                            'join_type': 'inner',
                            'on': join_key
                        })
                        result = platform.run_join(data['df'], data2, join_config)
                        num_records = len(result) if result is not None else 0
                    else:
                        # No suitable join key, skip join
                        result = pd.DataFrame()
                        num_records = 0
            
            elif experiment_type == 'time_window':
                if 'df' in data and 'timestamp' in data['df'].columns:
                    # Window aggregation - only use numeric columns
                    df = data['df'].copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Select only numeric columns for aggregation
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'timestamp' in numeric_cols:
                        numeric_cols.remove('timestamp')
                    
                    if len(numeric_cols) > 0:
                        # Use only numeric columns for resampling
                        df_numeric = df[['timestamp'] + numeric_cols].set_index('timestamp')
                        windowed = df_numeric.resample('1h').mean()
                        result = windowed.reset_index()
                        num_records = len(result)
                    else:
                        # No numeric columns, create simple time-based result
                        df_indexed = df.set_index('timestamp')
                        windowed = df_indexed.resample('1h').size().reset_index(name='count')
                        result = windowed
                        num_records = len(result)
            
            elif experiment_type == 'vector_knn':
                if 'vectors' in data:
                    k = config.get('k', 10)
                    query_vectors = data['vectors'][:10]  # Use first 10 as queries
                    result = platform.run_vector_search(data['vectors'], query_vectors, {'k': k})
                    num_records = len(query_vectors) * k
            
            elif experiment_type == 'text_similarity':
                if 'texts' in data:
                    k = config.get('k', 10)
                    query = data['texts'][0]  # Use first text as query
                    result = platform.run_text_similarity(data['texts'], query, {'k': k})
                    num_records = len(result)
            
            # End measurement
            perf_metrics = self.metrics_collector.end_measurement(num_records)
            
            metrics.latency_ms = perf_metrics['latency_ms']
            metrics.cpu_time_s = perf_metrics['cpu_time_s']
            metrics.memory_mb = perf_metrics['memory_mb']
            metrics.throughput = perf_metrics['throughput']
            metrics.is_correct = result is not None
            
            # Get agent reasoning
            if agent:
                metrics.decision_reasoning = agent.get_decision_reasoning()
            
        except Exception as e:
            logger.error(f"Error in experiment {experiment_id}: {e}")
            metrics.is_correct = False
            metrics.error_message = str(e)
            metrics.latency_ms = 0.0
        
        return metrics
    
    def run_all_experiments(self, agents: List[str], platforms: List[str],
                           data_sources: List[str], experiment_types: List[str],
                           config: Dict[str, Any] = None):
        """Run all combinations of experiments."""
        logger.info("Starting comprehensive experiment suite...")
        
        # Initialize agent manager with config if not already done
        if self.agent_manager is None:
            self.agent_manager = AgentManager(config)
            self.config = config
        
        # Check for quick mode - only run agent-selected platforms
        quick_mode = config.get('experiment', {}).get('quick_mode', False) if config else False
        
        if quick_mode:
            logger.info("QUICK MODE: Running only agent-selected platforms (much faster)")
            # In quick mode, we only run the platform each agent selects
            total_experiments = len(agents) * len(data_sources) * len(experiment_types)
        else:
            # Full mode: run all platform combinations
            total_experiments = len(agents) * len(platforms) * len(data_sources) * len(experiment_types)
        
        logger.info(f"Total experiments to run: {total_experiments}")
        
        # Pre-load all data once to avoid repeated I/O
        logger.info("Pre-loading all data sources...")
        for data_source in data_sources:
            self.load_data(data_source, use_cache=True)
        logger.info("Data pre-loading complete")
        
        # First pass: collect all results for learning agents
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            for agent_name in agents:
                agent = self.agent_manager.get_agent(agent_name)
                if not agent:
                    continue
                
                for data_source in data_sources:
                    # Load data once to get context
                    data = self.load_data(data_source)
                    context = {
                        'size': data.get('size', 0),
                        'num_columns': data.get('num_columns', 0),
                        'type': data.get('type', '')
                    }
                    
                    for experiment_type in experiment_types:
                        available_platforms = [p for p in platforms if self.platform_manager.is_available(p)]
                        
                        # Agent selects platform (for decision tracking)
                        selected_platform = agent.select_platform(
                            data_source, experiment_type, available_platforms, context
                        )
                        
                        # Determine which platforms to test
                        if quick_mode:
                            # Quick mode: only test the platform the agent selected
                            platforms_to_test = [selected_platform]
                        else:
                            # Full mode: test all platforms
                            platforms_to_test = available_platforms
                        
                        # Run experiment for selected platforms
                        for platform_name in platforms_to_test:
                            metrics = self.run_experiment(
                                agent_name, platform_name, data_source, experiment_type, config
                            )
                            
                            # Store metrics
                            self.all_metrics.append(metrics)
                            
                            # Update agent with results (for learning agents) - only for selected platform
                            if platform_name == selected_platform:
                                if agent_name == 'bandit':
                                    # Will update after we have all results
                                    pass
                                elif agent_name == 'cost_model':
                                    # Use agent's feature extraction method
                                    features = agent._extract_features(data_source, experiment_type, context)
                                    agent.train_model(platform_name, data_source, experiment_type,
                                                    features, metrics.latency_ms)
                                elif agent_name == 'hybrid':
                                    # Use cost model's feature extraction
                                    features = agent.cost_agent._extract_features(data_source, experiment_type, context)
                                    agent.update_cost_model(platform_name, data_source, experiment_type,
                                                          features, metrics.latency_ms)
                            
                            pbar.update(1)
                        
                        # Update bandit after all platforms tested
                        if agent_name == 'bandit':
                            # Find best latency for this experiment
                            experiment_metrics = [m for m in self.all_metrics 
                                                 if m.data_source == data_source and 
                                                 m.experiment_type == experiment_type and
                                                 m.agent == agent_name]
                            if experiment_metrics:
                                best_latency = min([m.latency_ms for m in experiment_metrics])
                                selected_metrics = [m for m in experiment_metrics 
                                                   if m.platform == selected_platform]
                                if selected_metrics:
                                    agent.update_reward(selected_platform, 
                                                       selected_metrics[0].latency_ms, 
                                                       best_latency)
                        
                        # Store decision trace (once per agent-data-experiment combo)
                        # Get timestamp from last metric
                        last_metric = [m for m in self.all_metrics 
                                      if m.agent == agent_name and 
                                      m.data_source == data_source and 
                                      m.experiment_type == experiment_type]
                        timestamp = last_metric[-1].timestamp if last_metric else ""
                        
                        decision_entry = {
                            'agent': agent_name,
                            'data_source': data_source,
                            'experiment_type': experiment_type,
                            'selected_platform': selected_platform,
                            'available_platforms': available_platforms,
                            'reasoning': agent.get_decision_reasoning(),
                            'timestamp': timestamp
                        }
                        self.decisions_trace.append(decision_entry)
        
        # Save raw metrics
        self._save_metrics()
        
        logger.info(f"Completed {len(self.all_metrics)} experiments")
    
    def _save_metrics(self):
        """Save all metrics to CSV files."""
        # Raw metrics
        raw_df = pd.DataFrame([m.to_dict() for m in self.all_metrics])
        raw_df.to_csv(self.output_dir / 'metrics_raw.csv', index=False)
        
        # Metrics by agent
        agent_df = raw_df.groupby('agent').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'throughput': ['mean', 'std'],
            'memory_mb': ['mean', 'std'],
            'is_correct': 'mean'
        }).reset_index()
        agent_df.columns = ['_'.join(col).strip('_') for col in agent_df.columns]
        agent_df.to_csv(self.output_dir / 'metrics_agents.csv', index=False)
        
        # Metrics by platform
        platform_df = raw_df.groupby('platform').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'throughput': ['mean', 'std'],
            'memory_mb': ['mean', 'std'],
            'is_correct': 'mean'
        }).reset_index()
        platform_df.columns = ['_'.join(col).strip('_') for col in platform_df.columns]
        platform_df.to_csv(self.output_dir / 'metrics_platforms.csv', index=False)
        
        # Metrics by data source
        datasource_df = raw_df.groupby('data_source').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'throughput': ['mean', 'std'],
            'memory_mb': ['mean', 'std'],
            'is_correct': 'mean'
        }).reset_index()
        datasource_df.columns = ['_'.join(col).strip('_') for col in datasource_df.columns]
        datasource_df.to_csv(self.output_dir / 'metrics_datasources.csv', index=False)
        
        # Decisions trace
        with open(self.output_dir / 'decisions_trace.json', 'w') as f:
            json.dump(self.decisions_trace, f, indent=2, default=str)
        
        logger.info("Saved all metrics files")

