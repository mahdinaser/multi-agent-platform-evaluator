"""
Thompson Sampling agent - Bayesian approach to platform selection.
Uses posterior distributions to balance exploration and exploitation.
"""
import logging
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class ThompsonAgent:
    """Thompson Sampling agent using Beta distributions."""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.name = "thompson"
        self.decision_history = []
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Per-platform Beta distribution parameters
        # (platform, data_source, experiment_type) -> {'alpha': float, 'beta': float, 'samples': []}
        self.platform_distributions = defaultdict(lambda: {'alpha': alpha_prior, 'beta': beta_prior, 'samples': []})
        
        # Success threshold (latency below this is "success")
        self.success_threshold_ms = 50.0
        self.adaptive_threshold = True
    
    def _get_key(self, platform: str, data_source: str, experiment_type: str) -> tuple:
        """Get key for platform distribution."""
        return (platform, data_source, experiment_type)
    
    def _sample_theta(self, platform: str, data_source: str, experiment_type: str) -> float:
        """Sample success probability from posterior Beta distribution."""
        key = self._get_key(platform, data_source, experiment_type)
        dist = self.platform_distributions[key]
        
        # Sample from Beta(alpha, beta)
        theta = np.random.beta(dist['alpha'], dist['beta'])
        return theta
    
    def _update_threshold(self):
        """Adaptively update success threshold based on observed latencies."""
        if not self.adaptive_threshold:
            return
        
        all_latencies = []
        for dist in self.platform_distributions.values():
            all_latencies.extend(dist['samples'])
        
        if len(all_latencies) >= 10:
            # Set threshold to median latency
            self.success_threshold_ms = np.median(all_latencies)
            logger.debug(f"Thompson updated threshold to {self.success_threshold_ms:.2f}ms")
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform using Thompson Sampling."""
        if not available_platforms:
            return None
        
        # Sample theta (success probability) for each platform
        sampled_thetas = {}
        for platform in available_platforms:
            theta = self._sample_theta(platform, data_source, experiment_type)
            sampled_thetas[platform] = theta
        
        # Select platform with highest sampled theta
        selected = max(sampled_thetas, key=sampled_thetas.get)
        
        # Get current distribution parameters
        key = self._get_key(selected, data_source, experiment_type)
        dist = self.platform_distributions[key]
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'Thompson: {selected} (theta={sampled_thetas[selected]:.3f})',
            'sampled_thetas': sampled_thetas,
            'alpha': dist['alpha'],
            'beta': dist['beta'],
            'success_threshold_ms': self.success_threshold_ms
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Update Beta distribution based on observed latency."""
        latency = metrics.get('latency_ms', float('inf'))
        
        # Determine if this is a "success" (latency below threshold)
        success = latency < self.success_threshold_ms
        
        # Update distribution
        key = self._get_key(platform, data_source, experiment_type)
        dist = self.platform_distributions[key]
        
        if success:
            dist['alpha'] += 1  # Increment successes
        else:
            dist['beta'] += 1   # Increment failures
        
        dist['samples'].append(latency)
        
        # Periodically update threshold
        if len(dist['samples']) % 10 == 0:
            self._update_threshold()
        
        logger.debug(f"Thompson updated {platform}: latency={latency:.2f}ms, "
                    f"success={success}, alpha={dist['alpha']:.1f}, beta={dist['beta']:.1f}")
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Thompson Sampling statistics."""
        total_successes = sum(d['alpha'] - self.alpha_prior for d in self.platform_distributions.values())
        total_failures = sum(d['beta'] - self.beta_prior for d in self.platform_distributions.values())
        total_samples = total_successes + total_failures
        
        return {
            'contexts_tracked': len(self.platform_distributions),
            'total_samples': int(total_samples),
            'total_successes': int(total_successes),
            'total_failures': int(total_failures),
            'success_rate': total_successes / total_samples if total_samples > 0 else 0,
            'success_threshold_ms': self.success_threshold_ms,
            'alpha_prior': self.alpha_prior,
            'beta_prior': self.beta_prior
        }

