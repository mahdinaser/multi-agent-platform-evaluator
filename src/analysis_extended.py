"""
Extended analysis methods for deeper insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def add_extended_analysis(analysis_engine):
    """Add extended analysis methods to AnalysisEngine."""
    
    def _summary_performance_rankings(self):
        """Rank platforms and agents by performance metrics."""
        df = self.metrics_df
        
        # Platform rankings
        platform_rankings = []
        for metric in ['latency_ms', 'throughput', 'memory_mb', 'cpu_time_s']:
            if metric in df.columns:
                if metric == 'latency_ms' or metric == 'memory_mb' or metric == 'cpu_time_s':
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
        
        # Agent rankings
        agent_rankings = []
        for metric in ['latency_ms', 'throughput']:
            if metric in df.columns:
                if metric == 'latency_ms':
                    ranked = df.groupby('agent')[metric].mean().sort_values().reset_index()
                    ranked['rank'] = range(1, len(ranked) + 1)
                else:
                    ranked = df.groupby('agent')[metric].mean().sort_values(ascending=False).reset_index()
                    ranked['rank'] = range(1, len(ranked) + 1)
                ranked['metric'] = metric
                agent_rankings.append(ranked)
        
        if agent_rankings:
            agent_rank_df = pd.concat(agent_rankings, ignore_index=True)
            agent_rank_df.to_csv(self.output_dir / 'summary_agent_rankings.csv', index=False)
    
    def _summary_correlations(self):
        """Calculate correlations between metrics."""
        df = self.metrics_df
        
        numeric_cols = ['latency_ms', 'throughput', 'memory_mb', 'cpu_time_s']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
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
                    subset = agent_df[(agent_df['data_source'] == data_source) & 
                                     (agent_df['experiment_type'] == exp_type)]
                    
                    if len(subset) > 0:
                        # Find best platform for this scenario
                        all_scenario = df[(df['data_source'] == data_source) & 
                                         (df['experiment_type'] == exp_type)]
                        best_latency = all_scenario['latency_ms'].min()
                        agent_latency = subset['latency_ms'].mean()
                        
                        effectiveness.append({
                            'agent': agent,
                            'data_source': data_source,
                            'experiment_type': exp_type,
                            'avg_latency_ms': agent_latency,
                            'best_latency_ms': best_latency,
                            'efficiency_ratio': best_latency / agent_latency if agent_latency > 0 else 0,
                            'is_optimal': agent_latency == best_latency
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
                scenario_df = df[(df['data_source'] == data_source) & 
                                (df['experiment_type'] == exp_type)]
                
                if len(scenario_df) > 0:
                    # Best by latency
                    best_latency = scenario_df.loc[scenario_df['latency_ms'].idxmin()]
                    # Best by throughput
                    if 'throughput' in scenario_df.columns:
                        best_throughput = scenario_df.loc[scenario_df['throughput'].idxmax()]
                    else:
                        best_throughput = None
                    # Best by memory
                    if 'memory_mb' in scenario_df.columns:
                        best_memory = scenario_df.loc[scenario_df['memory_mb'].idxmin()]
                    else:
                        best_memory = None
                    
                    recommendations.append({
                        'data_source': data_source,
                        'experiment_type': exp_type,
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
        except ImportError:
            logger.warning("scipy not available, skipping statistical tests")
            pd.DataFrame().to_csv(self.output_dir / 'summary_statistical_tests.csv', index=False)
            return
        
        results = []
        
        # Compare platforms
        platforms = df['platform'].unique()
        if len(platforms) >= 2:
            for i, p1 in enumerate(platforms):
                for p2 in platforms[i+1:]:
                    p1_data = df[df['platform'] == p1]['latency_ms'].values
                    p2_data = df[df['platform'] == p2]['latency_ms'].values
                    
                    if len(p1_data) > 0 and len(p2_data) > 0:
                        try:
                            stat, p_value = stats.mannwhitneyu(p1_data, p2_data, alternative='two-sided')
                            results.append({
                                'test': 'platform_comparison',
                                'group1': p1,
                                'group2': p2,
                                'statistic': stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
                        except:
                            pass
        
        # Compare agents
        agents = df['agent'].unique()
        if len(agents) >= 2:
            for i, a1 in enumerate(agents):
                for a2 in agents[i+1:]:
                    a1_data = df[df['agent'] == a1]['latency_ms'].values
                    a2_data = df[df['agent'] == a2]['latency_ms'].values
                    
                    if len(a1_data) > 0 and len(a2_data) > 0:
                        try:
                            stat, p_value = stats.mannwhitneyu(a1_data, a2_data, alternative='two-sided')
                            results.append({
                                'test': 'agent_comparison',
                                'group1': a1,
                                'group2': a2,
                                'statistic': stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
                        except:
                            pass
        
        if results:
            stats_df = pd.DataFrame(results)
            stats_df.to_csv(self.output_dir / 'summary_statistical_tests.csv', index=False)
        else:
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
            
            # Efficiency score (lower is better): weighted combination
            efficiency = (avg_latency / 1000.0) + (avg_memory / 100.0) + (avg_cpu * 10.0)
            
            cost_benefit.append({
                'platform': platform,
                'avg_latency_ms': avg_latency,
                'avg_memory_mb': avg_memory,
                'avg_cpu_time_s': avg_cpu,
                'efficiency_score': efficiency,
                'latency_per_memory': avg_latency / avg_memory if avg_memory > 0 else 0
            })
        
        cb_df = pd.DataFrame(cost_benefit)
        cb_df = cb_df.sort_values('efficiency_score')
        cb_df.to_csv(self.output_dir / 'summary_cost_benefit.csv', index=False)
    
    # Add methods to the class
    analysis_engine._summary_performance_rankings = _summary_performance_rankings.__get__(analysis_engine, type(analysis_engine))
    analysis_engine._summary_correlations = _summary_correlations.__get__(analysis_engine, type(analysis_engine))
    analysis_engine._summary_agent_effectiveness = _summary_agent_effectiveness.__get__(analysis_engine, type(analysis_engine))
    analysis_engine._summary_platform_recommendations = _summary_platform_recommendations.__get__(analysis_engine, type(analysis_engine))
    analysis_engine._summary_statistical_tests = _summary_statistical_tests.__get__(analysis_engine, type(analysis_engine))
    analysis_engine._summary_cost_benefit = _summary_cost_benefit.__get__(analysis_engine, type(analysis_engine))

