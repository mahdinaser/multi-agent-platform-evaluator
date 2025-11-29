"""
Bandit agent using UCB1 algorithm.
"""
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import math

logger = logging.getLogger(__name__)

class BanditAgent:
    """Agent using UCB1 multi-armed bandit algorithm."""
    
    def __init__(self):
        self.name = "bandit"
        self.platform_rewards = {}  # platform -> list of rewards
        self.platform_counts = {}  # platform -> number of times selected
        self.total_pulls = 0
        self.decision_history = []
        self.regret_history = []
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str],
                       context: Dict[str, Any] = None) -> str:
        """Select platform using UCB1."""
        if context is None:
            context = {}
        
        # Initialize platforms if not seen
        for platform in available_platforms:
            if platform not in self.platform_rewards:
                self.platform_rewards[platform] = []
                self.platform_counts[platform] = 0
        
        # UCB1 selection
        best_platform = None
        best_ucb = -float('inf')
        
        for platform in available_platforms:
            count = self.platform_counts[platform]
            
            if count == 0:
                # Never tried this platform - explore
                ucb = float('inf')
            else:
                # Calculate average reward
                avg_reward = np.mean(self.platform_rewards[platform]) if self.platform_rewards[platform] else 0
                
                # UCB1 formula
                ucb = avg_reward + math.sqrt(2 * math.log(self.total_pulls + 1) / count)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_platform = platform
        
        decision = best_platform or available_platforms[0]
        self.platform_counts[decision] += 1
        self.total_pulls += 1
        
        reasoning = f"UCB1: selected {decision} (UCB={best_ucb:.3f}, count={self.platform_counts[decision]})"
        
        self.decision_history.append({
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': decision,
            'reasoning': reasoning,
            'ucb_value': best_ucb,
            'context': context
        })
        
        return decision
    
    def update_reward(self, platform: str, reward: float, best_possible_reward: float = None):
        """Update reward for a platform."""
        if platform not in self.platform_rewards:
            self.platform_rewards[platform] = []
            self.platform_counts[platform] = 0
        
        # Normalize reward (assume latency-based, so lower is better)
        # Convert to reward: 1 / (1 + latency_ms / 1000)
        normalized_reward = 1.0 / (1.0 + reward / 1000.0)
        
        self.platform_rewards[platform].append(normalized_reward)
        
        # Calculate regret if best possible reward is known
        if best_possible_reward is not None:
            regret = best_possible_reward - normalized_reward
            self.regret_history.append({
                'platform': platform,
                'regret': regret,
                'reward': normalized_reward
            })
    
    def get_decision_reasoning(self) -> str:
        """Get reasoning for last decision."""
        if self.decision_history:
            return self.decision_history[-1].get('reasoning', '')
        return ""
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get decision history."""
        return self.decision_history
    
    def get_regret_curve(self) -> List[float]:
        """Get cumulative regret over time."""
        if not self.regret_history:
            return []
        return [r['regret'] for r in self.regret_history]

