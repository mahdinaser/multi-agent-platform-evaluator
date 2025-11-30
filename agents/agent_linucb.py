"""
LinUCB (Contextual Bandit) agent - uses features to make context-aware decisions.
Better than UCB1 because it incorporates workload characteristics.
"""
import logging
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class LinucbAgent:
    """Contextual bandit agent using Linear Upper Confidence Bound."""
    
    def __init__(self, alpha: float = 0.5, feature_dim: int = 10):
        self.name = "linucb"
        self.decision_history = []
        self.alpha = alpha  # Exploration parameter
        self.feature_dim = feature_dim
        
        # Per-platform LinUCB parameters
        self.platforms_params = {}  # platform -> {'A': matrix, 'b': vector}
        
        self.context_encodings = {
            'data_source': {},
            'experiment_type': {}
        }
        self.next_encoding_id = 0
    
    def _get_encoding_id(self, category: str, value: str) -> int:
        """Get or create encoding ID for a categorical value."""
        if value not in self.context_encodings[category]:
            self.context_encodings[category][value] = self.next_encoding_id
            self.next_encoding_id += 1
        return self.context_encodings[category][value]
    
    def _extract_features(self, data_source: str, experiment_type: str, context: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from context."""
        # Feature vector: [bias, data_source_id, experiment_type_id, data_size_log, ...]
        features = np.zeros(self.feature_dim)
        
        # Bias term
        features[0] = 1.0
        
        # One-hot encoding for data source (modulo feature_dim)
        ds_id = self._get_encoding_id('data_source', data_source)
        features[1 + (ds_id % (self.feature_dim // 2))] = 1.0
        
        # One-hot encoding for experiment type
        exp_id = self._get_encoding_id('experiment_type', experiment_type)
        features[1 + (self.feature_dim // 2) + (exp_id % (self.feature_dim // 2 - 1))] = 1.0
        
        # Additional context features if available
        if context:
            data_size = context.get('data_size', 0)
            if data_size > 0:
                features[self.feature_dim - 1] = np.log10(data_size + 1) / 10.0  # Normalized log size
        
        return features
    
    def _initialize_platform(self, platform: str):
        """Initialize LinUCB parameters for a platform."""
        if platform not in self.platforms_params:
            self.platforms_params[platform] = {
                'A': np.identity(self.feature_dim),  # A = I
                'b': np.zeros(self.feature_dim)      # b = 0
            }
    
    def _compute_ucb(self, platform: str, features: np.ndarray) -> float:
        """Compute UCB score for a platform given features."""
        params = self.platforms_params[platform]
        A = params['A']
        b = params['b']
        
        # Compute theta = A^{-1} * b
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # Singular matrix, add small regularization
            A_inv = np.linalg.inv(A + 0.01 * np.identity(self.feature_dim))
        
        theta = A_inv @ b
        
        # Expected reward
        expected_reward = theta @ features
        
        # Confidence bonus
        confidence = self.alpha * np.sqrt(features @ A_inv @ features)
        
        ucb_score = expected_reward + confidence
        return ucb_score
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform using LinUCB."""
        if not available_platforms:
            return None
        
        context = context or {}
        
        # Extract feature vector
        features = self._extract_features(data_source, experiment_type, context)
        
        # Initialize platforms if needed
        for platform in available_platforms:
            self._initialize_platform(platform)
        
        # Compute UCB scores for all platforms
        ucb_scores = {}
        for platform in available_platforms:
            ucb_scores[platform] = self._compute_ucb(platform, features)
        
        # Select platform with highest UCB
        selected = max(ucb_scores, key=ucb_scores.get)
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'LinUCB: {selected} (UCB={ucb_scores[selected]:.3f})',
            'ucb_scores': ucb_scores,
            'features': features.tolist(),
            'alpha': self.alpha
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Update LinUCB parameters based on observed reward."""
        # Convert latency to reward (lower latency = higher reward)
        latency = metrics.get('latency_ms', 1000)
        # Reward = -log(latency), normalized
        reward = -np.log(max(latency, 0.1)) / 10.0
        
        # Extract features (need context from last decision)
        last_decision = None
        for decision in reversed(self.decision_history):
            if (decision['selected_platform'] == platform and
                decision['data_source'] == data_source and
                decision['experiment_type'] == experiment_type):
                last_decision = decision
                break
        
        if last_decision is None:
            logger.warning(f"LinUCB update: No matching decision found for {platform}, {data_source}, {experiment_type}")
            return
        
        features = np.array(last_decision['features'])
        
        # Update A and b
        params = self.platforms_params[platform]
        params['A'] += np.outer(features, features)
        params['b'] += reward * features
        
        logger.debug(f"LinUCB updated {platform}: reward={reward:.3f}, latency={latency:.2f}ms")
    
    def get_decision_reasoning(self) -> str:
        """Get the reasoning for the last decision."""
        if self.decision_history:
            last_decision = self.decision_history[-1]
            return last_decision.get('reasoning', 'No reasoning available')
        return 'No decisions made yet'
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get LinUCB statistics."""
        return {
            'platforms_tracked': len(self.platforms_params),
            'feature_dim': self.feature_dim,
            'alpha': self.alpha,
            'decisions_made': len(self.decision_history),
            'unique_data_sources': len(self.context_encodings['data_source']),
            'unique_experiment_types': len(self.context_encodings['experiment_type'])
        }

