"""
Random agent - selects platforms uniformly at random.
Serves as performance floor baseline.
"""
import random
from typing import Dict, Any, List

class RandomAgent:
    """Agent that randomly selects platforms."""
    
    def __init__(self):
        self.name = "random"
        self.decision_history = []
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Randomly select a platform."""
        if not available_platforms:
            return None
        
        selected = random.choice(available_platforms)
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': 'Random selection'
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Random agent doesn't learn, so update is a no-op."""
        pass
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history

