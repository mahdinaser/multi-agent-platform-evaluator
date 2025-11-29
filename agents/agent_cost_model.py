"""
Cost-model agent that predicts latency.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class CostModelAgent:
    """Agent that uses cost models to predict platform performance."""
    
    def __init__(self):
        self.name = "cost_model"
        self.models = {}  # (platform, data_source, experiment_type) -> model
        self.training_data = defaultdict(list)  # (platform, data_source, experiment_type) -> [(features, latency)]
        self.scalers = {}  # (platform, data_source, experiment_type) -> scaler
        self.decision_history = []
    
    def _extract_features(self, data_source: str, experiment_type: str, context: Dict[str, Any]) -> np.ndarray:
        """Extract features for cost prediction."""
        features = []
        
        # Data size
        size = context.get('size', 0)
        features.append(size)
        features.append(np.log1p(size))
        
        # Data source type (one-hot encoded as numeric)
        source_types = ['tabular', 'logs', 'vectors', 'timeseries', 'text']
        for st in source_types:
            features.append(1.0 if data_source.startswith(st) else 0.0)
        
        # Experiment type (one-hot encoded)
        exp_types = ['scan', 'filter', 'aggregate', 'join', 'time_window', 'vector_knn', 'text_similarity']
        for et in exp_types:
            features.append(1.0 if experiment_type == et else 0.0)
        
        # Additional context features
        features.append(context.get('num_columns', 0))
        features.append(context.get('num_groups', 0))
        features.append(context.get('k', 10))
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, platform: str, data_source: str, experiment_type: str,
                   features: np.ndarray, actual_latency: float):
        """Train cost model with new data point."""
        key = (platform, data_source, experiment_type)
        
        # Store training data
        self.training_data[key].append((features.flatten(), actual_latency))
        
        # Retrain model if we have enough data
        if len(self.training_data[key]) >= 3:
            X = np.array([f for f, _ in self.training_data[key]])
            y = np.array([l for _, l in self.training_data[key]])
            
            # Scale features
            if key not in self.scalers:
                self.scalers[key] = StandardScaler()
                X_scaled = self.scalers[key].fit_transform(X)
            else:
                X_scaled = self.scalers[key].transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            self.models[key] = model
    
    def predict_latency(self, platform: str, data_source: str, experiment_type: str,
                       context: Dict[str, Any]) -> Optional[float]:
        """Predict latency for a platform."""
        key = (platform, data_source, experiment_type)
        
        if key not in self.models:
            # No model yet - return None (will use default)
            return None
        
        features = self._extract_features(data_source, experiment_type, context)
        
        # Scale features
        if key in self.scalers:
            features_scaled = self.scalers[key].transform(features)
        else:
            features_scaled = features
        
        # Predict
        prediction = self.models[key].predict(features_scaled)[0]
        return max(0.0, prediction)  # Ensure non-negative
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str],
                       context: Dict[str, Any] = None) -> str:
        """Select platform with lowest predicted latency."""
        if context is None:
            context = {}
        
        predictions = {}
        reasoning_parts = []
        
        for platform in available_platforms:
            pred = self.predict_latency(platform, data_source, experiment_type, context)
            if pred is not None:
                predictions[platform] = pred
                reasoning_parts.append(f"{platform}: {pred:.2f}ms")
            else:
                # No model yet - use default estimate based on data size
                default_pred = context.get('size', 100000) / 1000.0
                predictions[platform] = default_pred
                reasoning_parts.append(f"{platform}: {default_pred:.2f}ms (default)")
        
        # Select platform with lowest predicted latency
        best_platform = min(predictions.keys(), key=lambda p: predictions[p])
        best_prediction = predictions[best_platform]
        
        reasoning = f"Cost model: {best_platform} (predicted {best_prediction:.2f}ms) | " + " | ".join(reasoning_parts)
        
        self.decision_history.append({
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': best_platform,
            'predicted_latency': best_prediction,
            'all_predictions': predictions,
            'reasoning': reasoning,
            'context': context
        })
        
        return best_platform
    
    def get_decision_reasoning(self) -> str:
        """Get reasoning for last decision."""
        if self.decision_history:
            return self.decision_history[-1].get('reasoning', '')
        return ""
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get decision history."""
        return self.decision_history

