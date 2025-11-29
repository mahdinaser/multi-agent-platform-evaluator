"""
Round-Robin agent - cycles through platforms fairly.
Ensures equal testing of all platforms.
"""
from typing import Dict, Any, List

class RoundRobinAgent:
    """Agent that cycles through platforms in round-robin fashion."""
    
    def __init__(self):
        self.name = "round_robin"
        self.decision_history = []
        self.decision_count = 0
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform in round-robin order."""
        if not available_platforms:
            return None
        
        # Select based on decision count modulo number of platforms
        selected = available_platforms[self.decision_count % len(available_platforms)]
        self.decision_count += 1
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'Round-robin selection (iteration {self.decision_count})'
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Round-robin doesn't learn, so update is a no-op."""
        pass
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history

