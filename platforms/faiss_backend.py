"""
FAISS backend for vector search operations.
"""
import faiss
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FAISSBackend:
    """FAISS-based vector search backend."""
    
    def __init__(self):
        self.name = "faiss"
        self.index = None
    
    def _build_index(self, vectors: np.ndarray):
        """Build FAISS index."""
        dimension = vectors.shape[1]
        # Use L2 distance index
        self.index = faiss.IndexFlatL2(dimension)
        # Convert to float32
        vectors_f32 = vectors.astype(np.float32)
        self.index.add(vectors_f32)
    
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
        """Vector similarity search using FAISS."""
        if config is None:
            config = {}
        k = config.get('k', 10)
        
        # Build index if not exists or vectors changed
        self._build_index(vectors)
        
        # Convert query to float32
        query_f32 = query_vectors.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_f32, k)
        
        # Convert to list of lists
        results = [indices[i].tolist() for i in range(len(query_vectors))]
        return results
    
    def run_text_similarity(self, texts: List[str], query: str, config: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Text similarity - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_text_similarity(texts, query, config)

