"""
Oracle agent - always selects the optimal platform.
Requires post-hoc knowledge (performance ceiling baseline).
"""
import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

class OracleAgent:
    """Agent that always selects the optimal platform (post-hoc)."""
    
    def __init__(self):
        self.name = "oracle"
        self.decision_history = []
        self.performance_cache = {}  # (data_source, experiment_type, platform) -> latency
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform with best known performance."""
        if not available_platforms:
            return None
        
        # Look up cached performance
        key = (data_source, experiment_type)
        platform_latencies = {}
        
        for platform in available_platforms:
            cache_key = (data_source, experiment_type, platform)
            if cache_key in self.performance_cache:
                platform_latencies[platform] = self.performance_cache[cache_key]
        
        if platform_latencies:
            # Select platform with minimum latency
            selected = min(platform_latencies, key=platform_latencies.get)
            reasoning = f"Oracle: {selected} has best latency ({platform_latencies[selected]:.2f} ms)"
        else:
            # No cached performance, select first available
            selected = available_platforms[0]
            reasoning = "Oracle: No cached performance, selecting first platform"
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': reasoning,
            'known_latencies': platform_latencies
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Cache observed performance."""
        latency = metrics.get('latency_ms', float('inf'))
        cache_key = (data_source, experiment_type, platform)
        self.performance_cache[cache_key] = latency
        logger.debug(f"Oracle cached: {cache_key} -> {latency:.2f} ms")
    
    def get_decision_reasoning(self) -> str:
        """Get the reasoning for the last decision."""
        if self.decision_history:
            last_decision = self.decision_history[-1]
            return last_decision.get('reasoning', 'No reasoning available')
        return 'No decisions made yet'
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history

