"""
Analysis module for generating summary tables.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Generates comprehensive analysis tables."""
    
    def __init__(self, metrics_file: str, output_dir: str):
        self.metrics_df = pd.read_csv(metrics_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_tables(self):
        """Generate all summary tables."""
        logger.info("Generating summary tables...")
        
        # 1. Overall summary
        self._summary_overall()
        
        # 2. By platform
        self._summary_by_platform()
        
        # 3. By agent
        self._summary_by_agent()
        
        # 4. By data source
        self._summary_by_datasource()
        
        # 5. Latency statistics
        self._summary_latency_stats()
        
        # 6. Memory statistics
        self._summary_memory_stats()
        
        # 7. CPU statistics
        self._summary_cpu_stats()
        
        # 8. Vector performance
        self._summary_vector_performance()
        
        # 9. Text similarity performance
        self._summary_text_similarity()
        
        # 10. Stability metrics
        self._summary_stability()
        
        # 11. Agent accuracy
        self._summary_agent_accuracy()
        
        # 12. Agent regret (for bandit)
        self._summary_agent_regret()
        
        # 13. Performance rankings
        self._summary_performance_rankings()
        
        # 14. Correlation analysis
        self._summary_correlations()
        
        # 15. Agent effectiveness by data type
        self._summary_agent_effectiveness()
        
        # 16. Platform recommendations
        self._summary_platform_recommendations()
        
        # 17. Statistical significance tests
        self._summary_statistical_tests()
        
        # 18. Cost-benefit analysis
        self._summary_cost_benefit()
        
        logger.info("All summary tables generated!")
    
    def _summary_overall(self):
        """Overall summary statistics."""
        df = self.metrics_df
        
        summary = {
            'total_experiments': [len(df)],
            'successful_experiments': [df['is_correct'].sum()],
            'success_rate': [df['is_correct'].mean()],
            'avg_latency_ms': [df['latency_ms'].mean()],
            'median_latency_ms': [df['latency_ms'].median()],
            'std_latency_ms': [df['latency_ms'].std()],
            'min_latency_ms': [df['latency_ms'].min()],
            'max_latency_ms': [df['latency_ms'].max()],
            'avg_throughput': [df['throughput'].mean()],
            'avg_memory_mb': [df['memory_mb'].mean()],
            'avg_cpu_time_s': [df['cpu_time_s'].mean()]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / 'summary_overall.csv', index=False)
    
    def _summary_by_platform(self):
        """Summary grouped by platform."""
        df = self.metrics_df
        
        summary = df.groupby('platform').agg({
            'latency_ms': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'throughput': ['mean', 'std'],
            'memory_mb': ['mean', 'std', 'max'],
            'cpu_time_s': ['mean', 'std'],
            'is_correct': 'mean'
        }).reset_index()
        
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary.to_csv(self.output_dir / 'summary_by_platform.csv', index=False)
    
    def _summary_by_agent(self):
        """Summary grouped by agent."""
        df = self.metrics_df
        
        summary = df.groupby('agent').agg({
            'latency_ms': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'throughput': ['mean', 'std'],
            'memory_mb': ['mean', 'std'],
            'cpu_time_s': ['mean', 'std'],
            'is_correct': 'mean'
        }).reset_index()
        
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary.to_csv(self.output_dir / 'summary_by_agent.csv', index=False)
    
    def _summary_by_datasource(self):
        """Summary grouped by data source."""
        df = self.metrics_df
        
        summary = df.groupby('data_source').agg({
            'latency_ms': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'throughput': ['mean', 'std'],
            'memory_mb': ['mean', 'std'],
            'cpu_time_s': ['mean', 'std'],
            'is_correct': 'mean'
        }).reset_index()
        
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
        summary.to_csv(self.output_dir / 'summary_by_datasource.csv', index=False)
    
    def _summary_latency_stats(self):
        """Detailed latency statistics."""
        df = self.metrics_df
        
        latency_stats = []
        for group_col in ['platform', 'agent', 'data_source', 'experiment_type']:
            if group_col in df.columns:
                grouped = df.groupby(group_col)['latency_ms']
                stats = grouped.agg(['mean', 'median', 'std', 'min', 'max', 
                                    lambda x: np.percentile(x, 25),
                                    lambda x: np.percentile(x, 75),
                                    lambda x: np.percentile(x, 90),
                                    lambda x: np.percentile(x, 95),
                                    lambda x: np.percentile(x, 99)]).reset_index()
                stats.columns = [group_col, 'mean', 'median', 'std', 'min', 'max', 
                                'p25', 'p75', 'p90', 'p95', 'p99']
                stats['group_by'] = group_col
                latency_stats.append(stats)
        
        if latency_stats:
            combined = pd.concat(latency_stats, ignore_index=True)
            combined.to_csv(self.output_dir / 'summary_latency_stats.csv', index=False)
    
    def _summary_memory_stats(self):
        """Memory usage statistics."""
        df = self.metrics_df
        
        memory_stats = df.groupby(['platform', 'data_source']).agg({
            'memory_mb': ['mean', 'std', 'min', 'max', 'median']
        }).reset_index()
        
        memory_stats.columns = ['_'.join(col).strip('_') for col in memory_stats.columns]
        memory_stats.to_csv(self.output_dir / 'summary_memory_stats.csv', index=False)
    
    def _summary_cpu_stats(self):
        """CPU time statistics."""
        df = self.metrics_df
        
        cpu_stats = df.groupby(['platform', 'experiment_type']).agg({
            'cpu_time_s': ['mean', 'std', 'min', 'max', 'median']
        }).reset_index()
        
        cpu_stats.columns = ['_'.join(col).strip('_') for col in cpu_stats.columns]
        cpu_stats.to_csv(self.output_dir / 'summary_cpu_stats.csv', index=False)
    
    def _summary_vector_performance(self):
        """Vector search performance."""
        df = self.metrics_df
        vector_df = df[df['experiment_type'] == 'vector_knn'].copy()
        
        if len(vector_df) > 0:
            vector_perf = vector_df.groupby(['platform', 'data_source']).agg({
                'latency_ms': ['mean', 'std', 'min', 'max'],
                'throughput': ['mean', 'std']
            }).reset_index()
            
            vector_perf.columns = ['_'.join(col).strip('_') for col in vector_perf.columns]
            vector_perf.to_csv(self.output_dir / 'summary_vector_performance.csv', index=False)
        else:
            # Create empty file
            pd.DataFrame().to_csv(self.output_dir / 'summary_vector_performance.csv', index=False)
    
    def _summary_text_similarity(self):
        """Text similarity performance."""
        df = self.metrics_df
        text_df = df[df['experiment_type'] == 'text_similarity'].copy()
        
        if len(text_df) > 0:
            text_perf = text_df.groupby(['platform', 'data_source']).agg({
                'latency_ms': ['mean', 'std', 'min', 'max'],
                'throughput': ['mean', 'std']
            }).reset_index()
            
            text_perf.columns = ['_'.join(col).strip('_') for col in text_perf.columns]
            text_perf.to_csv(self.output_dir / 'summary_text_similarity.csv', index=False)
        else:
            pd.DataFrame().to_csv(self.output_dir / 'summary_text_similarity.csv', index=False)
    
    def _summary_stability(self):
        """Stability metrics (variance, std dev)."""
        df = self.metrics_df
        
        # Calculate stability per platform-agent combination
        stability = df.groupby(['platform', 'agent', 'experiment_type']).agg({
            'latency_ms': ['mean', 'std', lambda x: x.std() / x.mean() if x.mean() > 0 else 0]
        }).reset_index()
        
        stability.columns = ['platform', 'agent', 'experiment_type', 'mean_latency', 'std_latency', 'cv']
        stability['stability_score'] = 1.0 / (1.0 + stability['cv'])
        stability.to_csv(self.output_dir / 'summary_stability.csv', index=False)
    
    def _summary_agent_accuracy(self):
        """Agent decision accuracy."""
        df = self.metrics_df
        
        # For each agent, calculate how often they selected the best platform
        agent_accuracy = []
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent].copy()
            
            # For each experiment type and data source, find best platform
            for exp_type in agent_df['experiment_type'].unique():
                for data_source in agent_df['data_source'].unique():
                    subset = agent_df[(agent_df['experiment_type'] == exp_type) & 
                                     (agent_df['data_source'] == data_source)]
                    
                    if len(subset) > 0:
                        best_latency = subset['latency_ms'].min()
                        best_platform = subset.loc[subset['latency_ms'].idxmin(), 'platform']
                        agent_selected = subset.iloc[0]['platform']
                        
                        agent_accuracy.append({
                            'agent': agent,
                            'experiment_type': exp_type,
                            'data_source': data_source,
                            'best_platform': best_platform,
                            'best_latency_ms': best_latency,
                            'agent_selected': agent_selected,
                            'agent_latency_ms': subset.iloc[0]['latency_ms'],
                            'is_optimal': agent_selected == best_platform
                        })
        
        accuracy_df = pd.DataFrame(agent_accuracy)
        if len(accuracy_df) > 0:
            accuracy_summary = accuracy_df.groupby('agent').agg({
                'is_optimal': 'mean',
                'agent_latency_ms': 'mean',
                'best_latency_ms': 'mean'
            }).reset_index()
            accuracy_summary['accuracy'] = accuracy_summary['is_optimal']
            accuracy_summary['latency_ratio'] = accuracy_summary['agent_latency_ms'] / accuracy_summary['best_latency_ms']
            accuracy_summary.to_csv(self.output_dir / 'summary_agent_accuracy.csv', index=False)
        else:
            pd.DataFrame().to_csv(self.output_dir / 'summary_agent_accuracy.csv', index=False)
    
    def _summary_agent_regret(self):
        """Agent regret metrics (for bandit agent)."""
        df = self.metrics_df
        
        # Calculate regret for bandit agent
        bandit_df = df[df['agent'] == 'bandit'].copy()
        
        if len(bandit_df) > 0:
            regret_data = []
            for exp_type in bandit_df['experiment_type'].unique():
                for data_source in bandit_df['data_source'].unique():
                    subset = bandit_df[(bandit_df['experiment_type'] == exp_type) & 
                                      (bandit_df['data_source'] == data_source)]
                    
                    if len(subset) > 0:
                        best_latency = subset['latency_ms'].min()
                        for _, row in subset.iterrows():
                            regret = row['latency_ms'] - best_latency
                            regret_data.append({
                                'experiment_type': exp_type,
                                'data_source': data_source,
                                'platform': row['platform'],
                                'latency_ms': row['latency_ms'],
                                'best_latency_ms': best_latency,
                                'regret_ms': regret
                            })
            
            regret_df = pd.DataFrame(regret_data)
            if len(regret_df) > 0:
                regret_summary = regret_df.groupby('platform').agg({
                    'regret_ms': ['mean', 'std', 'sum', 'max']
                }).reset_index()
                regret_summary.columns = ['platform', 'mean_regret', 'std_regret', 'total_regret', 'max_regret']
                regret_summary.to_csv(self.output_dir / 'summary_agent_regret.csv', index=False)
            else:
                pd.DataFrame().to_csv(self.output_dir / 'summary_agent_regret.csv', index=False)
        else:
            pd.DataFrame().to_csv(self.output_dir / 'summary_agent_regret.csv', index=False)
    
    def _summary_performance_rankings(self):
        """Rank platforms and agents by performance metrics."""
        df = self.metrics_df
        
        # Platform rankings
        platform_rankings = []
        for metric in ['latency_ms', 'throughput', 'memory_mb', 'cpu_time_s']:
            if metric in df.columns:
                if metric in ['latency_ms', 'memory_mb', 'cpu_time_s']:
                    # Lower is better
                    ranked = df.groupby('platform')[metric].mean().sort_values().reset_index()
                    ranked['rank'] = range(1, len(ranked) + 1)
                    ranked['metric'] = metric
                    ranked['direction'] = 'lower_better'
                else:
                    # Higher is better
                    ranked = df.groupby('platform')[metric].mean().sort_values(ascending=False).reset_index()
                    ranked['rank'] = range(1, len(ranked) + 1)
                    ranked['metric'] = metric
                    ranked['direction'] = 'higher_better'
                platform_rankings.append(ranked)
        
        if platform_rankings:
            platform_rank_df = pd.concat(platform_rankings, ignore_index=True)
            platform_rank_df.to_csv(self.output_dir / 'summary_performance_rankings.csv', index=False)
    
    def _summary_correlations(self):
        """Calculate correlations between metrics."""
        df = self.metrics_df
        numeric_cols = [col for col in ['latency_ms', 'throughput', 'memory_mb', 'cpu_time_s'] if col in df.columns]
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            corr_matrix.to_csv(self.output_dir / 'summary_correlations.csv')
    
    def _summary_agent_effectiveness(self):
        """Agent effectiveness by data type and experiment type."""
        df = self.metrics_df
        effectiveness = []
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            for data_source in agent_df['data_source'].unique():
                for exp_type in agent_df['experiment_type'].unique():
                    subset = agent_df[(agent_df['data_source'] == data_source) & (agent_df['experiment_type'] == exp_type)]
                    if len(subset) > 0:
                        all_scenario = df[(df['data_source'] == data_source) & (df['experiment_type'] == exp_type)]
                        best_latency = all_scenario['latency_ms'].min()
                        agent_latency = subset['latency_ms'].mean()
                        effectiveness.append({
                            'agent': agent, 'data_source': data_source, 'experiment_type': exp_type,
                            'avg_latency_ms': agent_latency, 'best_latency_ms': best_latency,
                            'efficiency_ratio': best_latency / agent_latency if agent_latency > 0 else 0,
                            'is_optimal': abs(agent_latency - best_latency) < 0.01
                        })
        eff_df = pd.DataFrame(effectiveness)
        if len(eff_df) > 0:
            eff_df.to_csv(self.output_dir / 'summary_agent_effectiveness.csv', index=False)
        else:
            pd.DataFrame().to_csv(self.output_dir / 'summary_agent_effectiveness.csv', index=False)
    
    def _summary_platform_recommendations(self):
        """Generate platform recommendations by scenario."""
        df = self.metrics_df
        recommendations = []
        for data_source in df['data_source'].unique():
            for exp_type in df['experiment_type'].unique():
                scenario_df = df[(df['data_source'] == data_source) & (df['experiment_type'] == exp_type)]
                if len(scenario_df) > 0:
                    best_latency = scenario_df.loc[scenario_df['latency_ms'].idxmin()]
                    best_throughput = scenario_df.loc[scenario_df['throughput'].idxmax()] if 'throughput' in scenario_df.columns else None
                    best_memory = scenario_df.loc[scenario_df['memory_mb'].idxmin()] if 'memory_mb' in scenario_df.columns else None
                    recommendations.append({
                        'data_source': data_source, 'experiment_type': exp_type,
                        'best_latency_platform': best_latency['platform'],
                        'best_latency_ms': best_latency['latency_ms'],
                        'best_throughput_platform': best_throughput['platform'] if best_throughput is not None else None,
                        'best_throughput': best_throughput['throughput'] if best_throughput is not None else None,
                        'best_memory_platform': best_memory['platform'] if best_memory is not None else None,
                        'best_memory_mb': best_memory['memory_mb'] if best_memory is not None else None
                    })
        rec_df = pd.DataFrame(recommendations)
        if len(rec_df) > 0:
            rec_df.to_csv(self.output_dir / 'summary_platform_recommendations.csv', index=False)
        else:
            pd.DataFrame().to_csv(self.output_dir / 'summary_platform_recommendations.csv', index=False)
    
    def _summary_statistical_tests(self):
        """Perform statistical significance tests."""
        df = self.metrics_df
        try:
            from scipy import stats
            results = []
            platforms = df['platform'].unique()
            if len(platforms) >= 2:
                for i, p1 in enumerate(platforms):
                    for p2 in platforms[i+1:]:
                        p1_data = df[df['platform'] == p1]['latency_ms'].values
                        p2_data = df[df['platform'] == p2]['latency_ms'].values
                        if len(p1_data) > 0 and len(p2_data) > 0:
                            try:
                                stat, p_value = stats.mannwhitneyu(p1_data, p2_data, alternative='two-sided')
                                results.append({'test': 'platform_comparison', 'group1': p1, 'group2': p2,
                                               'statistic': stat, 'p_value': p_value, 'significant': p_value < 0.05})
                            except: pass
            if results:
                pd.DataFrame(results).to_csv(self.output_dir / 'summary_statistical_tests.csv', index=False)
            else:
                pd.DataFrame().to_csv(self.output_dir / 'summary_statistical_tests.csv', index=False)
        except ImportError:
            logger.warning("scipy not available, skipping statistical tests")
            pd.DataFrame().to_csv(self.output_dir / 'summary_statistical_tests.csv', index=False)
    
    def _summary_cost_benefit(self):
        """Cost-benefit analysis: latency vs memory trade-offs."""
        df = self.metrics_df
        cost_benefit = []
        for platform in df['platform'].unique():
            platform_df = df[df['platform'] == platform]
            avg_latency = platform_df['latency_ms'].mean()
            avg_memory = platform_df['memory_mb'].mean() if 'memory_mb' in platform_df.columns else 0
            avg_cpu = platform_df['cpu_time_s'].mean() if 'cpu_time_s' in platform_df.columns else 0
            efficiency = (avg_latency / 1000.0) + (avg_memory / 100.0) + (avg_cpu * 10.0)
            cost_benefit.append({
                'platform': platform, 'avg_latency_ms': avg_latency, 'avg_memory_mb': avg_memory,
                'avg_cpu_time_s': avg_cpu, 'efficiency_score': efficiency,
                'latency_per_memory': avg_latency / avg_memory if avg_memory > 0 else 0
            })
        cb_df = pd.DataFrame(cost_benefit).sort_values('efficiency_score')
        cb_df.to_csv(self.output_dir / 'summary_cost_benefit.csv', index=False)

