"""
DuckDB backend for analytical SQL operations.
"""
import duckdb
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DuckDBBackend:
    """DuckDB-based analytical backend."""
    
    def __init__(self):
        self.name = "duckdb"
        self.conn = duckdb.connect()
    
    def _df_to_table(self, data: pd.DataFrame, table_name: str = "data"):
        """Register DataFrame as DuckDB table."""
        self.conn.register(table_name, data)
        return table_name
    
    def run_scan(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Full table scan."""
        table_name = self._df_to_table(data, "scan_data")
        result = self.conn.execute(f"SELECT * FROM {table_name}").df()
        return result
    
    def run_filter(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter operation."""
        predicate = config.get('predicate', {})
        table_name = self._df_to_table(data, "filter_data")
        
        if not predicate:
            return self.run_scan(data)
        
        col = predicate.get('column')
        op = predicate.get('operator', '==')
        value = predicate.get('value')
        
        # Handle different value types for DuckDB
        if isinstance(value, str):
            # Properly quote strings and timestamps
            value = f"'{value}'"
        elif pd.api.types.is_datetime64_any_dtype(type(value)):
            # Convert datetime to string for DuckDB
            value = f"TIMESTAMP '{str(value)}'"
        elif hasattr(value, 'strftime'):
            # Handle pandas Timestamp objects
            value = f"TIMESTAMP '{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        
        sql = f"SELECT * FROM {table_name} WHERE {col} {op} {value}"
        result = self.conn.execute(sql).df()
        return result
    
    def run_aggregate(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Group-by aggregation."""
        group_cols = config.get('group_by', [])
        agg_funcs = config.get('aggregations', {})
        table_name = self._df_to_table(data, "agg_data")
        
        # DuckDB function mapping (DuckDB uses AVG instead of MEAN)
        func_map = {
            'mean': 'AVG',
            'avg': 'AVG',
            'sum': 'SUM',
            'count': 'COUNT',
            'min': 'MIN',
            'max': 'MAX',
            'std': 'STDDEV',  # DuckDB has STDDEV
            'var': 'VARIANCE'  # DuckDB has VARIANCE
        }
        
        if not group_cols:
            # Global aggregation
            agg_parts = []
            for col, func in agg_funcs.items():
                if col in data.columns:
                    sql_func = func_map.get(func.lower(), func.upper())
                    agg_parts.append(f"{sql_func}({col}) as {col}_{func}")
            
            sql = f"SELECT {', '.join(agg_parts)} FROM {table_name}"
        else:
            # Group by
            group_str = ', '.join(group_cols)
            agg_parts = []
            for col, func in agg_funcs.items():
                if col in data.columns:
                    sql_func = func_map.get(func.lower(), func.upper())
                    agg_parts.append(f"{sql_func}({col}) as {col}_{func}")
            
            sql = f"SELECT {group_str}, {', '.join(agg_parts)} FROM {table_name} GROUP BY {group_str}"
        
        result = self.conn.execute(sql).df()
        return result
    
    def run_join(self, data1: pd.DataFrame, data2: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Join operation."""
        join_type = config.get('join_type', 'inner').upper()
        on = config.get('on', None)
        left_on = config.get('left_on', None)
        right_on = config.get('right_on', None)
        
        if on:
            left_on = right_on = on
        
        table1 = self._df_to_table(data1, "join_data1")
        table2 = self._df_to_table(data2, "join_data2")
        
        sql = f"SELECT * FROM {table1} {join_type} JOIN {table2} ON {table1}.{left_on} = {table2}.{right_on}"
        result = self.conn.execute(sql).df()
        return result
    
    def run_vector_search(self, vectors: np.ndarray, query_vectors: np.ndarray, config: Dict[str, Any] = None) -> List[List[int]]:
        """Vector search - fallback to pandas method."""
        # DuckDB has vector support but for simplicity, use pandas method
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_vector_search(vectors, query_vectors, config)
    
    def run_text_similarity(self, texts: List[str], query: str, config: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Text similarity - fallback to pandas method."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_text_similarity(texts, query, config)

