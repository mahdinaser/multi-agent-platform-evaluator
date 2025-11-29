"""
Metrics tracking for experiments.
"""
import time
import psutil
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    experiment_id: str
    agent: str
    platform: str
    data_source: str
    experiment_type: str
    
    # Performance metrics
    latency_ms: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0
    cpu_time_s: float = 0.0
    
    # Correctness
    is_correct: bool = True
    error_message: Optional[str] = None
    
    # Stability
    variance: float = 0.0
    std_dev: float = 0.0
    
    # Agent-specific
    predicted_latency: Optional[float] = None
    actual_latency: Optional[float] = None
    reward: Optional[float] = None
    regret: Optional[float] = None
    decision_reasoning: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'agent': self.agent,
            'platform': self.platform,
            'data_source': self.data_source,
            'experiment_type': self.experiment_type,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'memory_mb': self.memory_mb,
            'cpu_time_s': self.cpu_time_s,
            'is_correct': self.is_correct,
            'error_message': self.error_message,
            'variance': self.variance,
            'std_dev': self.std_dev,
            'predicted_latency': self.predicted_latency,
            'actual_latency': self.actual_latency,
            'reward': self.reward,
            'regret': self.regret,
            'decision_reasoning': self.decision_reasoning,
            'timestamp': self.timestamp,
            'config': json.dumps(self.config) if self.config else None
        }

class MetricsCollector:
    """Collect performance metrics during experiments."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def start_measurement(self):
        """Start measuring resources."""
        self.start_time = time.time()
        self.start_cpu = self.process.cpu_times()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def end_measurement(self, num_records: int = 1) -> Dict[str, float]:
        """End measurement and return metrics."""
        end_time = time.time()
        end_cpu = self.process.cpu_times()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        latency_ms = (end_time - self.start_time) * 1000
        cpu_time_s = (end_cpu.user + end_cpu.system) - (self.start_cpu.user + self.start_cpu.system)
        memory_mb = end_memory - self.start_memory
        throughput = num_records / (end_time - self.start_time) if (end_time - self.start_time) > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'cpu_time_s': cpu_time_s,
            'memory_mb': memory_mb,
            'throughput': throughput
        }
    
    def measure_stability(self, latencies: list) -> Dict[str, float]:
        """Calculate stability metrics from multiple runs."""
        if not latencies:
            return {'variance': 0.0, 'std_dev': 0.0}
        
        import numpy as np
        latencies_array = np.array(latencies)
        variance = float(np.var(latencies_array))
        std_dev = float(np.std(latencies_array))
        
        return {
            'variance': variance,
            'std_dev': std_dev
        }

