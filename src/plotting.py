"""
Plotting module for generating visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class PlottingEngine:
    """Generates comprehensive visualizations."""
    
    def __init__(self, metrics_file: str, output_dir: str):
        self.metrics_df = pd.read_csv(metrics_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        logger.info("Generating visualizations...")
        
        # 1. Latency distribution per platform
        self._plot_latency_distribution()
        
        # 2. Memory usage comparison
        self._plot_memory_comparison()
        
        # 3. CPU time comparison
        self._plot_cpu_comparison()
        
        # 4. Stability heatmap
        self._plot_stability_heatmap()
        
        # 5. Agent decision frequency
        self._plot_agent_decisions()
        
        # 6. Regret curves (bandit)
        self._plot_regret_curves()
        
        # 7. LLM agent confusion matrix
        self._plot_llm_confusion_matrix()
        
        # 8. Accuracy vs latency scatter
        self._plot_accuracy_latency_scatter()
        
        # 9. Platform ranking radar chart
        self._plot_platform_radar()
        
        # 10. End-to-end comparison
        self._plot_e2e_comparison()
        
        # 11. Platform performance by experiment type
        self._plot_platform_experiment_heatmap()
        
        # 12. Agent learning curves
        self._plot_agent_learning_curves()
        
        # 13. Latency vs throughput scatter
        self._plot_latency_throughput_scatter()
        
        # 14. Platform efficiency (latency/memory)
        self._plot_platform_efficiency()
        
        # 15. Agent decision patterns
        self._plot_agent_decision_patterns()
        
        # 16. Performance distribution violin plots
        self._plot_performance_distributions()
        
        # 17. Correlation heatmap
        self._plot_correlation_heatmap()
        
        # 18. Best platform per scenario
        self._plot_best_platform_scenarios()
        
        logger.info("All visualizations generated!")
    
    def _plot_latency_distribution(self):
        """Latency distribution per platform."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        platforms = self.metrics_df['platform'].unique()
        data_to_plot = [self.metrics_df[self.metrics_df['platform'] == p]['latency_ms'].values 
                       for p in platforms]
        
        ax.boxplot(data_to_plot, labels=platforms)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_xlabel('Platform', fontsize=12)
        ax.set_title('Latency Distribution per Platform', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_comparison(self):
        """Memory usage comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        memory_by_platform = self.metrics_df.groupby('platform')['memory_mb'].mean().sort_values()
        
        ax.barh(memory_by_platform.index, memory_by_platform.values, color='steelblue')
        ax.set_xlabel('Average Memory Usage (MB)', fontsize=12)
        ax.set_ylabel('Platform', fontsize=12)
        ax.set_title('Memory Usage Comparison Across Platforms', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cpu_comparison(self):
        """CPU time comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cpu_by_platform = self.metrics_df.groupby(['platform', 'experiment_type'])['cpu_time_s'].mean().reset_index()
        
        pivot = cpu_by_platform.pivot(index='platform', columns='experiment_type', values='cpu_time_s')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_ylabel('CPU Time (seconds)', fontsize=12)
        ax.set_xlabel('Platform', fontsize=12)
        ax.set_title('CPU Time Comparison by Experiment Type', fontsize=14, fontweight='bold')
        ax.legend(title='Experiment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cpu_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_heatmap(self):
        """Stability heatmap."""
        stability = self.metrics_df.groupby(['platform', 'agent'])['latency_ms'].agg(['mean', 'std']).reset_index()
        stability['cv'] = stability['std'] / stability['mean']
        stability['stability'] = 1.0 / (1.0 + stability['cv'])
        
        pivot = stability.pivot(index='platform', columns='agent', values='stability')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Stability Score'})
        ax.set_title('Stability Heatmap (Platform × Agent)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_decisions(self):
        """Agent decision frequency bar chart."""
        decision_counts = self.metrics_df.groupby(['agent', 'platform']).size().reset_index(name='count')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        agents = decision_counts['agent'].unique()
        platforms = decision_counts['platform'].unique()
        x = np.arange(len(agents))
        width = 0.15
        
        for i, platform in enumerate(platforms):
            counts = [decision_counts[(decision_counts['agent'] == a) & 
                                     (decision_counts['platform'] == platform)]['count'].sum() 
                     for a in agents]
            ax.bar(x + i * width, counts, width, label=platform)
        
        ax.set_ylabel('Decision Count', fontsize=12)
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_title('Agent Decision Frequency by Platform', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(platforms) - 1) / 2)
        ax.set_xticklabels(agents)
        ax.legend(title='Platform')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'agent_decisions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regret_curves(self):
        """Regret curves for bandit agent."""
        bandit_df = self.metrics_df[self.metrics_df['agent'] == 'bandit'].copy()
        
        if len(bandit_df) > 0:
            # Calculate cumulative regret
            bandit_df = bandit_df.sort_values('timestamp')
            bandit_df['best_latency'] = bandit_df.groupby(['experiment_type', 'data_source'])['latency_ms'].transform('min')
            bandit_df['regret'] = bandit_df['latency_ms'] - bandit_df['best_latency']
            bandit_df['cumulative_regret'] = bandit_df['regret'].cumsum()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(range(len(bandit_df)), bandit_df['cumulative_regret'], linewidth=2, color='darkred')
            ax.set_xlabel('Experiment Number', fontsize=12)
            ax.set_ylabel('Cumulative Regret (ms)', fontsize=12)
            ax.set_title('Bandit Agent Cumulative Regret Curve', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'regret_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No bandit agent data available', 
                   ha='center', va='center', fontsize=14)
            plt.savefig(self.output_dir / 'regret_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_llm_confusion_matrix(self):
        """LLM agent decision confusion matrix."""
        llm_df = self.metrics_df[self.metrics_df['agent'] == 'llm'].copy()
        
        if len(llm_df) > 0:
            # For each experiment, find best platform and compare with LLM choice
            confusion_data = []
            for exp_type in llm_df['experiment_type'].unique():
                for data_source in llm_df['data_source'].unique():
                    subset = llm_df[(llm_df['experiment_type'] == exp_type) & 
                                   (llm_df['data_source'] == data_source)]
                    if len(subset) > 0:
                        best_platform = subset.loc[subset['latency_ms'].idxmin(), 'platform']
                        llm_platform = subset.iloc[0]['platform']
                        confusion_data.append({
                            'best': best_platform,
                            'llm_selected': llm_platform
                        })
            
            if confusion_data:
                confusion_df = pd.DataFrame(confusion_data)
                confusion_matrix = pd.crosstab(confusion_df['best'], confusion_df['llm_selected'])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('LLM Selected Platform', fontsize=12)
                ax.set_ylabel('Best Platform', fontsize=12)
                ax.set_title('LLM Agent Decision Confusion Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'llm_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.text(0.5, 0.5, 'Insufficient data for confusion matrix', 
                       ha='center', va='center', fontsize=14)
                plt.savefig(self.output_dir / 'llm_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No LLM agent data available', 
                   ha='center', va='center', fontsize=14)
            plt.savefig(self.output_dir / 'llm_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_accuracy_latency_scatter(self):
        """Accuracy vs latency scatter plot."""
        # Calculate accuracy per agent
        agent_stats = []
        for agent in self.metrics_df['agent'].unique():
            agent_df = self.metrics_df[self.metrics_df['agent'] == agent]
            
            # Find best platform for each experiment
            accuracy = 0
            total = 0
            for exp_type in agent_df['experiment_type'].unique():
                for data_source in agent_df['data_source'].unique():
                    subset = agent_df[(agent_df['experiment_type'] == exp_type) & 
                                     (agent_df['data_source'] == data_source)]
                    if len(subset) > 0:
                        best_platform = subset.loc[subset['latency_ms'].idxmin(), 'platform']
                        agent_platform = subset.iloc[0]['platform']
                        if agent_platform == best_platform:
                            accuracy += 1
                        total += 1
            
            avg_latency = agent_df['latency_ms'].mean()
            agent_stats.append({
                'agent': agent,
                'accuracy': accuracy / total if total > 0 else 0,
                'avg_latency_ms': avg_latency
            })
        
        stats_df = pd.DataFrame(agent_stats)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(stats_df['avg_latency_ms'], stats_df['accuracy'], s=200, alpha=0.6)
        
        for _, row in stats_df.iterrows():
            ax.annotate(row['agent'], (row['avg_latency_ms'], row['accuracy']), 
                       fontsize=10, ha='center')
        
        ax.set_xlabel('Average Latency (ms)', fontsize=12)
        ax.set_ylabel('Accuracy (Optimal Selection Rate)', fontsize=12)
        ax.set_title('Agent Accuracy vs Average Latency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_latency_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_platform_radar(self):
        """Platform ranking radar chart."""
        # Calculate normalized scores for different metrics
        platforms = self.metrics_df['platform'].unique()
        
        metrics = {
            'Latency (lower better)': self.metrics_df.groupby('platform')['latency_ms'].mean(),
            'Throughput (higher better)': self.metrics_df.groupby('platform')['throughput'].mean(),
            'Memory Efficiency': 1.0 / (1.0 + self.metrics_df.groupby('platform')['memory_mb'].mean() / 1000),
            'CPU Efficiency': 1.0 / (1.0 + self.metrics_df.groupby('platform')['cpu_time_s'].mean()),
            'Success Rate': self.metrics_df.groupby('platform')['is_correct'].mean()
        }
        
        # Normalize each metric to 0-1 scale
        normalized = {}
        for metric_name, values in metrics.items():
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                normalized[metric_name] = (values - min_val) / (max_val - min_val)
            else:
                normalized[metric_name] = values * 0 + 0.5
        
        # Create radar chart for first few platforms
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(normalized), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for platform in list(platforms)[:5]:  # Limit to 5 platforms
            values = [normalized[m].get(platform, 0.5) for m in normalized.keys()]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=platform)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(normalized.keys()))
        ax.set_ylim(0, 1)
        ax.set_title('Platform Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_e2e_comparison(self):
        """End-to-end comparison summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Latency by platform and experiment type
        latency_pivot = self.metrics_df.groupby(['platform', 'experiment_type'])['latency_ms'].mean().reset_index()
        latency_pivot = latency_pivot.pivot(index='platform', columns='experiment_type', values='latency_ms')
        sns.heatmap(latency_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Average Latency (ms)', fontweight='bold')
        
        # 2. Throughput comparison
        throughput_by_platform = self.metrics_df.groupby('platform')['throughput'].mean().sort_values()
        axes[0, 1].barh(throughput_by_platform.index, throughput_by_platform.values, color='steelblue')
        axes[0, 1].set_title('Average Throughput', fontweight='bold')
        axes[0, 1].set_xlabel('Records/second')
        
        # 3. Success rate by agent
        success_by_agent = self.metrics_df.groupby('agent')['is_correct'].mean().sort_values()
        axes[1, 0].bar(success_by_agent.index, success_by_agent.values, color='green', alpha=0.7)
        axes[1, 0].set_title('Success Rate by Agent', fontweight='bold')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Memory vs Latency scatter
        platform_avg = self.metrics_df.groupby('platform').agg({
            'memory_mb': 'mean',
            'latency_ms': 'mean'
        }).reset_index()
        axes[1, 1].scatter(platform_avg['memory_mb'], platform_avg['latency_ms'], s=200, alpha=0.6)
        for _, row in platform_avg.iterrows():
            axes[1, 1].annotate(row['platform'], (row['memory_mb'], row['latency_ms']), fontsize=9)
        axes[1, 1].set_xlabel('Memory Usage (MB)')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Memory vs Latency Trade-off', fontweight='bold')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        
        plt.suptitle('End-to-End Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'e2e_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_platform_experiment_heatmap(self):
        """Heatmap of platform performance by experiment type."""
        pivot = self.metrics_df.groupby(['platform', 'experiment_type'])['latency_ms'].mean().reset_index()
        pivot = pivot.pivot(index='platform', columns='experiment_type', values='latency_ms')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Latency (ms)'})
        ax.set_title('Platform Performance by Experiment Type (Latency)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_experiment_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_learning_curves(self):
        """Learning curves for agents over time."""
        df = self.metrics_df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Latency over time
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            if len(agent_df) > 1:
                agent_df = agent_df.sort_values('timestamp' if 'timestamp' in agent_df.columns else 'experiment_id')
                axes[0].plot(range(len(agent_df)), agent_df['latency_ms'].rolling(5, min_periods=1).mean(), 
                            label=agent, linewidth=2)
        
        axes[0].set_xlabel('Experiment Number', fontsize=12)
        axes[0].set_ylabel('Average Latency (ms)', fontsize=12)
        axes[0].set_title('Agent Learning Curves - Latency', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy over time (if available)
        if 'is_correct' in df.columns:
            for agent in df['agent'].unique():
                agent_df = df[df['agent'] == agent]
                if len(agent_df) > 1:
                    agent_df = agent_df.sort_values('timestamp' if 'timestamp' in agent_df.columns else 'experiment_id')
                    axes[1].plot(range(len(agent_df)), agent_df['is_correct'].rolling(5, min_periods=1).mean(), 
                                label=agent, linewidth=2)
        
        axes[1].set_xlabel('Experiment Number', fontsize=12)
        axes[1].set_ylabel('Success Rate', fontsize=12)
        axes[1].set_title('Agent Learning Curves - Success Rate', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'agent_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_throughput_scatter(self):
        """Scatter plot of latency vs throughput."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for platform in self.metrics_df['platform'].unique():
            platform_df = self.metrics_df[self.metrics_df['platform'] == platform]
            ax.scatter(platform_df['latency_ms'], platform_df['throughput'], 
                      label=platform, s=100, alpha=0.6)
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Throughput (records/sec)', fontsize=12)
        ax.set_title('Latency vs Throughput Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_throughput_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_platform_efficiency(self):
        """Platform efficiency: latency per unit memory."""
        platform_eff = self.metrics_df.groupby('platform').agg({
            'latency_ms': 'mean',
            'memory_mb': 'mean',
            'cpu_time_s': 'mean'
        }).reset_index()
        
        platform_eff['efficiency'] = platform_eff['latency_ms'] / (platform_eff['memory_mb'] + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Efficiency bar chart
        platform_eff = platform_eff.sort_values('efficiency')
        axes[0].barh(platform_eff['platform'], platform_eff['efficiency'], color='steelblue')
        axes[0].set_xlabel('Efficiency (Latency/Memory)', fontsize=12)
        axes[0].set_title('Platform Efficiency Score', fontsize=14, fontweight='bold')
        
        # 3D scatter: latency, memory, CPU
        scatter = axes[1].scatter(platform_eff['memory_mb'], platform_eff['latency_ms'], 
                                 s=platform_eff['cpu_time_s']*1000, alpha=0.6, c=range(len(platform_eff)), cmap='viridis')
        for i, row in platform_eff.iterrows():
            axes[1].annotate(row['platform'], (row['memory_mb'], row['latency_ms']), fontsize=9)
        axes[1].set_xlabel('Memory (MB)', fontsize=12)
        axes[1].set_ylabel('Latency (ms)', fontsize=12)
        axes[1].set_title('Platform Resource Trade-offs\n(Bubble size = CPU time)', fontsize=14, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_agent_decision_patterns(self):
        """Analyze agent decision patterns."""
        # Load decisions trace if available
        decisions_file = self.output_dir.parent / 'decisions_trace.json'
        if decisions_file.exists():
            import json
            with open(decisions_file, 'r') as f:
                decisions = json.load(f)
            
            decisions_df = pd.DataFrame(decisions)
            
            # Decision patterns by agent and data source
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Platform selection frequency by agent
            if 'selected_platform' in decisions_df.columns:
                platform_counts = decisions_df.groupby(['agent', 'selected_platform']).size().reset_index(name='count')
                pivot = platform_counts.pivot(index='agent', columns='selected_platform', values='count').fillna(0)
                sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[0, 0])
                axes[0, 0].set_title('Platform Selection Frequency by Agent', fontweight='bold')
            
            # Decision by data source
            if 'data_source' in decisions_df.columns and 'selected_platform' in decisions_df.columns:
                data_platform = decisions_df.groupby(['data_source', 'selected_platform']).size().reset_index(name='count')
                pivot2 = data_platform.pivot(index='data_source', columns='selected_platform', values='count').fillna(0)
                sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='Greens', ax=axes[0, 1])
                axes[0, 1].set_title('Platform Selection by Data Source', fontweight='bold')
            
            # Experiment type preferences
            if 'experiment_type' in decisions_df.columns and 'selected_platform' in decisions_df.columns:
                exp_platform = decisions_df.groupby(['experiment_type', 'selected_platform']).size().reset_index(name='count')
                pivot3 = exp_platform.pivot(index='experiment_type', columns='selected_platform', values='count').fillna(0)
                sns.heatmap(pivot3, annot=True, fmt='.0f', cmap='Oranges', ax=axes[1, 0])
                axes[1, 0].set_title('Platform Selection by Experiment Type', fontweight='bold')
            
            # Agent consistency
            if 'agent' in decisions_df.columns and 'selected_platform' in decisions_df.columns:
                agent_consistency = decisions_df.groupby('agent')['selected_platform'].apply(
                    lambda x: x.value_counts().max() / len(x)
                ).reset_index(name='consistency')
                axes[1, 1].bar(agent_consistency['agent'], agent_consistency['consistency'], color='purple', alpha=0.7)
                axes[1, 1].set_title('Agent Decision Consistency', fontweight='bold')
                axes[1, 1].set_ylabel('Consistency Score')
                axes[1, 1].set_ylim(0, 1)
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'agent_decision_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Create empty plot if no decisions trace
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No decisions trace available', ha='center', va='center', fontsize=14)
            plt.savefig(self.output_dir / 'agent_decision_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_distributions(self):
        """Violin plots showing performance distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Latency distributions
        sns.violinplot(data=self.metrics_df, x='platform', y='latency_ms', ax=axes[0, 0])
        axes[0, 0].set_title('Latency Distribution by Platform', fontweight='bold')
        axes[0, 0].set_yscale('log')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Memory distributions
        if 'memory_mb' in self.metrics_df.columns:
            sns.violinplot(data=self.metrics_df, x='platform', y='memory_mb', ax=axes[0, 1])
            axes[0, 1].set_title('Memory Distribution by Platform', fontweight='bold')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Latency by agent
        sns.violinplot(data=self.metrics_df, x='agent', y='latency_ms', ax=axes[1, 0])
        axes[1, 0].set_title('Latency Distribution by Agent', fontweight='bold')
        axes[1, 0].set_yscale('log')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Throughput distributions
        if 'throughput' in self.metrics_df.columns:
            sns.violinplot(data=self.metrics_df, x='platform', y='throughput', ax=axes[1, 1])
            axes[1, 1].set_title('Throughput Distribution by Platform', fontweight='bold')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self):
        """Correlation heatmap of all metrics."""
        numeric_cols = ['latency_ms', 'throughput', 'memory_mb', 'cpu_time_s']
        numeric_cols = [col for col in numeric_cols if col in self.metrics_df.columns]
        
        if len(numeric_cols) > 1:
            corr_matrix = self.metrics_df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                       vmin=-1, vmax=1, ax=ax, square=True)
            ax.set_title('Metric Correlations', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_best_platform_scenarios(self):
        """Visualize best platform for each scenario."""
        best_platforms = []
        for data_source in self.metrics_df['data_source'].unique():
            for exp_type in self.metrics_df['experiment_type'].unique():
                scenario_df = self.metrics_df[(self.metrics_df['data_source'] == data_source) & 
                                              (self.metrics_df['experiment_type'] == exp_type)]
                if len(scenario_df) > 0:
                    best = scenario_df.loc[scenario_df['latency_ms'].idxmin()]
                    best_platforms.append({
                        'data_source': data_source,
                        'experiment_type': exp_type,
                        'best_platform': best['platform'],
                        'latency_ms': best['latency_ms']
                    })
        
        if best_platforms:
            best_df = pd.DataFrame(best_platforms)
            pivot = best_df.pivot(index='data_source', columns='experiment_type', values='best_platform')
            
            fig, ax = plt.subplots(figsize=(14, 8))
            # Create color mapping for platforms
            platforms = best_df['best_platform'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(platforms)))
            platform_colors = dict(zip(platforms, colors))
            
            # Create numeric matrix for heatmap
            platform_to_num = {p: i for i, p in enumerate(platforms)}
            pivot_num = pivot.applymap(lambda x: platform_to_num.get(x, -1))
            
            im = ax.imshow(pivot_num.values, cmap='Set3', aspect='auto')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
            ax.set_yticklabels(pivot.index)
            ax.set_title('Best Platform by Scenario', fontsize=14, fontweight='bold')
            
            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    text = pivot.iloc[i, j]
                    ax.text(j, i, text, ha='center', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'best_platform_scenarios.png', dpi=300, bbox_inches='tight')
            plt.close()

