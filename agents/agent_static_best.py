"""
Static-Best agent - selects the historically best platform.
Learns from initial warm-up phase, then sticks with best performer.
"""
import logging
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class StaticBestAgent:
    """Agent that selects historically best platform (after warm-up)."""
    
    def __init__(self, warmup_threshold: int = 20):
        self.name = "static_best"
        self.decision_history = []
        self.performance_history = defaultdict(list)  # platform -> [latencies]
        self.best_platform = None
        self.warmup_threshold = warmup_threshold
        self.decision_count = 0
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select best platform based on historical performance."""
        if not available_platforms:
            return None
        
        self.decision_count += 1
        
        # During warm-up or if no best determined yet, cycle through platforms
        if self.decision_count <= self.warmup_threshold or self.best_platform is None:
            # Round-robin during warm-up
            selected = available_platforms[self.decision_count % len(available_platforms)]
            reasoning = f"Warm-up phase ({self.decision_count}/{self.warmup_threshold}): exploring {selected}"
        else:
            # After warm-up, select best historical platform
            if self.best_platform in available_platforms:
                selected = self.best_platform
                reasoning = f"Post warm-up: using best platform {selected}"
            else:
                # Fallback if best not available
                selected = available_platforms[0]
                reasoning = f"Post warm-up: best {self.best_platform} not available, using {selected}"
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': reasoning,
            'warmup_complete': self.decision_count > self.warmup_threshold
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Update performance history and determine best platform."""
        latency = metrics.get('latency_ms', float('inf'))
        self.performance_history[platform].append(latency)
        
        # After warm-up threshold, determine best platform
        if self.decision_count == self.warmup_threshold:
            avg_latencies = {}
            for plat, latencies in self.performance_history.items():
                if latencies:
                    avg_latencies[plat] = sum(latencies) / len(latencies)
            
            if avg_latencies:
                self.best_platform = min(avg_latencies, key=avg_latencies.get)
                logger.info(f"Static-Best determined best platform: {self.best_platform} "
                           f"(avg latency: {avg_latencies[self.best_platform]:.2f} ms)")
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history

