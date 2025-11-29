"""
Hybrid agent combining multiple strategies.
"""
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Import other agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.agent_rule_based import RuleBasedAgent
from agents.agent_cost_model import CostModelAgent
from agents.agent_llm import LLMAgent

logger = logging.getLogger(__name__)

class HybridAgent:
    """Agent that combines rule-based, cost-model, and LLM agents."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.name = "hybrid"
        self.rule_agent = RuleBasedAgent()
        self.cost_agent = CostModelAgent()
        self.llm_agent = LLMAgent()
        
        # Default weights
        self.weights = weights or {
            'rule_based': 0.3,
            'cost_model': 0.5,
            'llm': 0.2
        }
        
        self.decision_history = []
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str],
                       context: Dict[str, Any] = None) -> str:
        """Select platform using weighted combination of agents."""
        if context is None:
            context = {}
        
        # Get recommendations from each agent
        rule_platform = self.rule_agent.select_platform(data_source, experiment_type, available_platforms, context)
        cost_platform = self.cost_agent.select_platform(data_source, experiment_type, available_platforms, context)
        llm_platform = self.llm_agent.select_platform(data_source, experiment_type, available_platforms, context)
        
        # Count votes with weights
        votes = {}
        votes[rule_platform] = votes.get(rule_platform, 0) + self.weights['rule_based']
        votes[cost_platform] = votes.get(cost_platform, 0) + self.weights['cost_model']
        votes[llm_platform] = votes.get(llm_platform, 0) + self.weights['llm']
        
        # Select platform with highest weighted vote
        decision = max(votes.keys(), key=lambda p: votes[p])
        
        reasoning = (
            f"Hybrid decision: {decision} "
            f"(rule: {rule_platform} [{self.weights['rule_based']:.2f}], "
            f"cost: {cost_platform} [{self.weights['cost_model']:.2f}], "
            f"llm: {llm_platform} [{self.weights['llm']:.2f}])"
        )
        
        self.decision_history.append({
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': decision,
            'rule_platform': rule_platform,
            'cost_platform': cost_platform,
            'llm_platform': llm_platform,
            'votes': votes,
            'reasoning': reasoning,
            'context': context
        })
        
        return decision
    
    def get_decision_reasoning(self) -> str:
        """Get reasoning for last decision."""
        if self.decision_history:
            return self.decision_history[-1].get('reasoning', '')
        return ""
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get decision history."""
        return self.decision_history
    
    def update_cost_model(self, platform: str, data_source: str, experiment_type: str,
                         features, actual_latency: float):
        """Update cost model with new data."""
        self.cost_agent.train_model(platform, data_source, experiment_type, features, actual_latency)

