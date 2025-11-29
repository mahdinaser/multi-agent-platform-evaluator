"""
Annoy backend for approximate nearest neighbor search.
"""
try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AnnoyBackend:
    """Annoy-based approximate nearest neighbor backend."""
    
    def __init__(self):
        self.name = "annoy"
        self.index = None
        if not ANNOY_AVAILABLE:
            logger.warning("Annoy not available, falling back to pandas for vector operations")
    
    def _build_index(self, vectors: np.ndarray):
        """Build Annoy index."""
        if not ANNOY_AVAILABLE:
            return
        
        dimension = vectors.shape[1]
        self.index = AnnoyIndex(dimension, 'angular')  # Angular distance for cosine similarity
        
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector.astype(np.float32))
        
        # Build with 10 trees
        self.index.build(10)
    
    def run_scan(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Full table scan - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_scan(data, config)
    
    def run_filter(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_filter(data, config)
    
    def run_aggregate(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_aggregate(data, config)
    
    def run_join(self, data1: pd.DataFrame, data2: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Join - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_join(data1, data2, config)
    
    def run_vector_search(self, vectors: np.ndarray, query_vectors: np.ndarray, config: Dict[str, Any] = None) -> List[List[int]]:
        """Vector similarity search using Annoy."""
        if config is None:
            config = {}
        k = config.get('k', 10)
        
        if not ANNOY_AVAILABLE:
            # Fallback to pandas
            import sys
            sys.path.insert(0, 'platforms')
            from pandas_backend import PandasBackend
            backend = PandasBackend()
            return backend.run_vector_search(vectors, query_vectors, config)
        
        # Build index
        self._build_index(vectors)
        
        # Search
        results = []
        for query in query_vectors:
            indices = self.index.get_nns_by_vector(query.astype(np.float32), k)
            results.append(indices)
        
        return results
    
    def run_text_similarity(self, texts: List[str], query: str, config: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Text similarity - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_text_similarity(texts, query, config)

