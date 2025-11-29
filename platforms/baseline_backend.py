"""
Baseline Python backend for naive implementations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class BaselineBackend:
    """Baseline naive Python implementation."""
    
    def __init__(self):
        self.name = "baseline"
    
    def run_scan(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Full table scan - naive iteration."""
        result = []
        for _, row in data.iterrows():
            result.append(row.to_dict())
        return pd.DataFrame(result)
    
    def run_filter(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter - naive iteration."""
        predicate = config.get('predicate', {})
        if not predicate:
            return self.run_scan(data)
        
        col = predicate.get('column')
        op = predicate.get('operator', '==')
        value = predicate.get('value')
        
        result = []
        for _, row in data.iterrows():
            row_val = row[col]
            match = False
            if op == '==' and row_val == value:
                match = True
            elif op == '>' and row_val > value:
                match = True
            elif op == '<' and row_val < value:
                match = True
            elif op == '>=' and row_val >= value:
                match = True
            elif op == '<=' and row_val <= value:
                match = True
            
            if match:
                result.append(row.to_dict())
        
        return pd.DataFrame(result)
    
    def run_aggregate(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate - naive groupby."""
        group_cols = config.get('group_by', [])
        agg_funcs = config.get('aggregations', {})
        
        if not group_cols:
            # Global aggregation
            result = {}
            for col, func in agg_funcs.items():
                if col in data.columns:
                    values = data[col].tolist()
                    if func == 'sum':
                        result[col] = [sum(values)]
                    elif func == 'mean':
                        result[col] = [sum(values) / len(values)]
                    elif func == 'count':
                        result[col] = [len(values)]
                    elif func == 'max':
                        result[col] = [max(values)]
                    elif func == 'min':
                        result[col] = [min(values)]
            return pd.DataFrame(result)
        
        # Group by - naive implementation
        groups = {}
        for _, row in data.iterrows():
            key = tuple(row[col] for col in group_cols)
            if key not in groups:
                groups[key] = []
            groups[key].append(row.to_dict())
        
        result_rows = []
        for key, group_rows in groups.items():
            row_dict = dict(zip(group_cols, key))
            for col, func in agg_funcs.items():
                if col in data.columns:
                    values = [r[col] for r in group_rows]
                    if func == 'sum':
                        row_dict[f'{col}_{func}'] = sum(values)
                    elif func == 'mean':
                        row_dict[f'{col}_{func}'] = sum(values) / len(values)
                    elif func == 'count':
                        row_dict[f'{col}_{func}'] = len(values)
                    elif func == 'max':
                        row_dict[f'{col}_{func}'] = max(values)
                    elif func == 'min':
                        row_dict[f'{col}_{func}'] = min(values)
            result_rows.append(row_dict)
        
        return pd.DataFrame(result_rows)
    
    def run_join(self, data1: pd.DataFrame, data2: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Join - naive nested loop."""
        join_type = config.get('join_type', 'inner')
        on = config.get('on', None)
        left_on = config.get('left_on', None)
        right_on = config.get('right_on', None)
        
        if on:
            left_on = right_on = on
        
        result = []
        for _, row1 in data1.iterrows():
            matched = False
            for _, row2 in data2.iterrows():
                if row1[left_on] == row2[right_on]:
                    merged = {**row1.to_dict(), **row2.to_dict()}
                    result.append(merged)
                    matched = True
            
            if join_type == 'left' and not matched:
                result.append(row1.to_dict())
        
        return pd.DataFrame(result)
    
    def run_vector_search(self, vectors: np.ndarray, query_vectors: np.ndarray, config: Dict[str, Any] = None) -> List[List[int]]:
        """Vector search - naive brute force."""
        if config is None:
            config = {}
        k = config.get('k', 10)
        
        results = []
        for query in query_vectors:
            # Compute distances to all vectors
            distances = []
            for i, vector in enumerate(vectors):
                dist = np.linalg.norm(query - vector)
                distances.append((i, dist))
            
            # Sort and get top-k
            distances.sort(key=lambda x: x[1])
            top_k = [idx for idx, _ in distances[:k]]
            results.append(top_k)
        
        return results
    
    def run_text_similarity(self, texts: List[str], query: str, config: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Text similarity - naive word overlap."""
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
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

