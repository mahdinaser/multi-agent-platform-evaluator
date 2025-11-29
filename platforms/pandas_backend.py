"""
Pandas backend for data operations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PandasBackend:
    """Pandas-based data processing backend."""
    
    def __init__(self):
        self.name = "pandas"
    
    def run_scan(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Full table scan."""
        if config is None:
            config = {}
        return data.copy()
    
    def run_filter(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter operation."""
        predicate = config.get('predicate', {})
        if not predicate:
            return data
        
        # Simple predicate: column, operator, value
        col = predicate.get('column')
        op = predicate.get('operator', '==')
        value = predicate.get('value')
        
        if col and col in data.columns:
            if op == '==':
                return data[data[col] == value]
            elif op == '>':
                return data[data[col] > value]
            elif op == '<':
                return data[data[col] < value]
            elif op == '>=':
                return data[data[col] >= value]
            elif op == '<=':
                return data[data[col] <= value]
        
        return data
    
    def run_aggregate(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Group-by aggregation."""
        group_cols = config.get('group_by', [])
        agg_funcs = config.get('aggregations', {})
        
        if not group_cols:
            # Global aggregation
            result = {}
            for col, func in agg_funcs.items():
                if col in data.columns:
                    if func == 'sum':
                        result[col] = [data[col].sum()]
                    elif func == 'mean':
                        result[col] = [data[col].mean()]
                    elif func == 'count':
                        result[col] = [len(data)]
                    elif func == 'max':
                        result[col] = [data[col].max()]
                    elif func == 'min':
                        result[col] = [data[col].min()]
            return pd.DataFrame(result)
        
        # Group by
        grouped = data.groupby(group_cols)
        result = grouped.agg(agg_funcs).reset_index()
        return result
    
    def run_join(self, data1: pd.DataFrame, data2: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Join operation."""
        join_type = config.get('join_type', 'inner')
        on = config.get('on', None)
        left_on = config.get('left_on', None)
        right_on = config.get('right_on', None)
        
        if on:
            left_on = right_on = on
        
        if join_type == 'inner':
            return pd.merge(data1, data2, left_on=left_on, right_on=right_on, how='inner')
        elif join_type == 'left':
            return pd.merge(data1, data2, left_on=left_on, right_on=right_on, how='left')
        elif join_type == 'right':
            return pd.merge(data1, data2, left_on=left_on, right_on=right_on, how='right')
        elif join_type == 'outer':
            return pd.merge(data1, data2, left_on=left_on, right_on=right_on, how='outer')
        
        return pd.DataFrame()
    
    def run_vector_search(self, vectors: np.ndarray, query_vectors: np.ndarray, config: Dict[str, Any] = None) -> List[List[int]]:
        """Vector similarity search using cosine similarity."""
        if config is None:
            config = {}
        k = config.get('k', 10)
        
        # Normalize vectors
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, vectors_norm.T)
        
        # Get top-k
        results = []
        for i in range(len(query_vectors)):
            top_k_indices = np.argsort(similarities[i])[::-1][:k]
            results.append(top_k_indices.tolist())
        
        return results
    
    def run_text_similarity(self, texts: List[str], query: str, config: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Simple text similarity using word overlap."""
        if config is None:
            config = {}
        k = config.get('k', 10)
        
        query_words = set(query.lower().split())
        similarities = []
        
        for i, text in enumerate(texts):
            text_words = set(text.lower().split())
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            similarity = intersection / union if union > 0 else 0.0
            similarities.append((i, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

