"""
Polars backend for fast DataFrame operations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("Polars not installed. Install with: pip install polars")

class PolarsBackend:
    """Polars-based fast DataFrame backend."""
    
    def __init__(self):
        self.name = "polars"
        if not POLARS_AVAILABLE:
            raise ImportError("Polars not available")
    
    def _pd_to_pl(self, data: pd.DataFrame) -> pl.DataFrame:
        """Convert pandas DataFrame to polars DataFrame."""
        return pl.from_pandas(data)
    
    def _pl_to_pd(self, data: pl.DataFrame) -> pd.DataFrame:
        """Convert polars DataFrame to pandas DataFrame."""
        return data.to_pandas()
    
    def run_scan(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Full table scan."""
        pl_df = self._pd_to_pl(data)
        # Simple select all
        result = pl_df.select(pl.all())
        return self._pl_to_pd(result)
    
    def run_filter(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter operation."""
        predicate = config.get('predicate', {})
        
        if not predicate:
            return self.run_scan(data)
        
        pl_df = self._pd_to_pl(data)
        col = predicate.get('column')
        op = predicate.get('operator', '==')
        value = predicate.get('value')
        
        # Build polars filter expression
        if op == '==':
            result = pl_df.filter(pl.col(col) == value)
        elif op == '>':
            result = pl_df.filter(pl.col(col) > value)
        elif op == '<':
            result = pl_df.filter(pl.col(col) < value)
        elif op == '>=':
            result = pl_df.filter(pl.col(col) >= value)
        elif op == '<=':
            result = pl_df.filter(pl.col(col) <= value)
        elif op == '!=':
            result = pl_df.filter(pl.col(col) != value)
        else:
            result = pl_df
        
        return self._pl_to_pd(result)
    
    def run_aggregate(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Group-by aggregation."""
        pl_df = self._pd_to_pl(data)
        group_by = config.get('group_by', [])
        aggregates = config.get('aggregates', {})
        
        if not group_by:
            # No grouping, just aggregates
            agg_exprs = []
            for col, agg_func in aggregates.items():
                if agg_func == 'sum':
                    agg_exprs.append(pl.col(col).sum().alias(f'{col}_sum'))
                elif agg_func == 'mean':
                    agg_exprs.append(pl.col(col).mean().alias(f'{col}_mean'))
                elif agg_func == 'count':
                    agg_exprs.append(pl.col(col).count().alias(f'{col}_count'))
            
            if agg_exprs:
                result = pl_df.select(agg_exprs)
            else:
                result = pl_df
        else:
            # Group by with aggregates
            agg_exprs = []
            for col, agg_func in aggregates.items():
                if agg_func == 'sum':
                    agg_exprs.append(pl.col(col).sum())
                elif agg_func == 'mean':
                    agg_exprs.append(pl.col(col).mean())
                elif agg_func == 'count':
                    agg_exprs.append(pl.col(col).count())
            
            if agg_exprs:
                result = pl_df.group_by(group_by).agg(agg_exprs)
            else:
                result = pl_df.group_by(group_by).count()
        
        return self._pl_to_pd(result)
    
    def run_join(self, data1: pd.DataFrame, data2: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Join operation."""
        pl_df1 = self._pd_to_pl(data1)
        pl_df2 = self._pd_to_pl(data2)
        
        join_type = config.get('join_type', 'inner')
        on = config.get('on', 'id')
        
        result = pl_df1.join(pl_df2, on=on, how=join_type)
        return self._pl_to_pd(result)
    
    def run_vector_search(self, vectors: np.ndarray, query_vectors: np.ndarray, 
                         config: Dict[str, Any]) -> List[np.ndarray]:
        """Vector similarity search (basic implementation)."""
        k = config.get('k', 10)
        results = []
        
        # Compute distances (brute force for simplicity)
        for query_vec in query_vectors:
            distances = np.linalg.norm(vectors - query_vec, axis=1)
            indices = np.argsort(distances)[:k]
            results.append(indices)
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        pass

