"""
SQLite backend for OLTP-style operations.
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

class SQLiteBackend:
    """SQLite-based backend."""
    
    def __init__(self):
        self.name = "sqlite"
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_file.close()
        self.conn = sqlite3.connect(self.db_file.name)
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'db_file') and os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)
    
    def _df_to_table(self, data: pd.DataFrame, table_name: str):
        """Store DataFrame as SQLite table."""
        data.to_sql(table_name, self.conn, if_exists='replace', index=False)
        return table_name
    
    def run_scan(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """Full table scan."""
        table_name = "scan_data"
        self._df_to_table(data, table_name)
        result = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        return result
    
    def run_filter(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter operation."""
        predicate = config.get('predicate', {})
        table_name = "filter_data"
        self._df_to_table(data, table_name)
        
        if not predicate:
            return self.run_scan(data)
        
        col = predicate.get('column')
        op = predicate.get('operator', '=')
        value = predicate.get('value')
        
        # Handle different value types for SQLite
        if isinstance(value, str):
            # Check if it's a timestamp-like string
            if ' ' in value and ':' in value:
                # SQLite timestamp - need to quote properly
                value = f"'{value}'"
            else:
                value = f"'{value}'"
        elif pd.api.types.is_datetime64_any_dtype(type(value)):
            # Convert datetime to string for SQLite
            value = f"'{str(value)}'"
        
        sql = f"SELECT * FROM {table_name} WHERE {col} {op} {value}"
        result = pd.read_sql_query(sql, self.conn)
        return result
    
    def run_aggregate(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Group-by aggregation."""
        group_cols = config.get('group_by', [])
        agg_funcs = config.get('aggregations', {})
        table_name = "agg_data"
        self._df_to_table(data, table_name)
        
        # SQLite function mapping (SQLite uses AVG instead of MEAN)
        func_map = {
            'mean': 'AVG',
            'avg': 'AVG',
            'sum': 'SUM',
            'count': 'COUNT',
            'min': 'MIN',
            'max': 'MAX',
            'std': 'AVG',  # SQLite doesn't have STD, fallback to AVG
            'var': 'AVG'   # SQLite doesn't have VAR, fallback to AVG
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
        
        result = pd.read_sql_query(sql, self.conn)
        return result
    
    def run_join(self, data1: pd.DataFrame, data2: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Join operation."""
        join_type = config.get('join_type', 'inner').upper()
        on = config.get('on', None)
        left_on = config.get('left_on', None)
        right_on = config.get('right_on', None)
        
        if on:
            left_on = right_on = on
        
        table1 = "join_data1"
        table2 = "join_data2"
        self._df_to_table(data1, table1)
        self._df_to_table(data2, table2)
        
        if join_type == 'INNER':
            sql = f"SELECT * FROM {table1} INNER JOIN {table2} ON {table1}.{left_on} = {table2}.{right_on}"
        elif join_type == 'LEFT':
            sql = f"SELECT * FROM {table1} LEFT JOIN {table2} ON {table1}.{left_on} = {table2}.{right_on}"
        elif join_type == 'RIGHT':
            sql = f"SELECT * FROM {table1} RIGHT JOIN {table2} ON {table1}.{left_on} = {table2}.{right_on}"
        else:
            sql = f"SELECT * FROM {table1} LEFT OUTER JOIN {table2} ON {table1}.{left_on} = {table2}.{right_on}"
        
        result = pd.read_sql_query(sql, self.conn)
        return result
    
    def run_vector_search(self, vectors: np.ndarray, query_vectors: np.ndarray, config: Dict[str, Any] = None) -> List[List[int]]:
        """Vector search - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_vector_search(vectors, query_vectors, config)
    
    def run_text_similarity(self, texts: List[str], query: str, config: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Text similarity - fallback to pandas."""
        import sys
        sys.path.insert(0, 'platforms')
        from pandas_backend import PandasBackend
        backend = PandasBackend()
        return backend.run_text_similarity(texts, query, config)

