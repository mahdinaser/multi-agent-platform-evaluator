"""
TPC-H benchmark data generator.
Generates TPC-H-like tables at different scale factors.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TPCHGenerator:
    """Generates TPC-H benchmark data."""
    
    def __init__(self, scale_factor: int = 1, seed: int = 42):
        """
        Initialize TPC-H generator.
        
        Args:
            scale_factor: Scale factor (1 = 1GB, 10 = 10GB, etc.)
            seed: Random seed for reproducibility
        """
        self.scale_factor = scale_factor
        self.seed = seed
        np.random.seed(seed)
        
        # TPC-H cardinalities at SF=1
        self.base_rows = {
            'customer': 150_000,
            'orders': 1_500_000,
            'lineitem': 6_000_000,
            'part': 200_000,
            'supplier': 10_000,
            'partsupp': 800_000,
            'nation': 25,
            'region': 5
        }
    
    def _generate_customer(self) -> pd.DataFrame:
        """Generate CUSTOMER table."""
        n_rows = self.base_rows['customer'] * self.scale_factor
        logger.info(f"Generating CUSTOMER table: {n_rows:,} rows")
        
        data = {
            'c_custkey': range(1, n_rows + 1),
            'c_name': [f'Customer#{i:09d}' for i in range(1, n_rows + 1)],
            'c_address': [f'Address {i}' for i in range(1, n_rows + 1)],
            'c_nationkey': np.random.randint(0, 25, n_rows),
            'c_phone': [f'{np.random.randint(10, 34)}-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' 
                       for _ in range(n_rows)],
            'c_acctbal': np.random.uniform(-999.99, 9999.99, n_rows).round(2),
            'c_mktsegment': np.random.choice(['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD'], n_rows),
            'c_comment': ['Comment text' for _ in range(n_rows)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_orders(self) -> pd.DataFrame:
        """Generate ORDERS table."""
        n_rows = self.base_rows['orders'] * self.scale_factor
        logger.info(f"Generating ORDERS table: {n_rows:,} rows")
        
        n_customers = self.base_rows['customer'] * self.scale_factor
        base_date = datetime(1992, 1, 1)
        
        data = {
            'o_orderkey': range(1, n_rows + 1),
            'o_custkey': np.random.randint(1, n_customers + 1, n_rows),
            'o_orderstatus': np.random.choice(['O', 'F', 'P'], n_rows, p=[0.5, 0.25, 0.25]),
            'o_totalprice': np.random.uniform(1000, 500000, n_rows).round(2),
            'o_orderdate': [(base_date + timedelta(days=int(np.random.randint(0, 2557)))).strftime('%Y-%m-%d') 
                           for _ in range(n_rows)],
            'o_orderpriority': np.random.choice(['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW'], n_rows),
            'o_clerk': [f'Clerk#{np.random.randint(1, 1001):09d}' for _ in range(n_rows)],
            'o_shippriority': np.random.randint(0, 2, n_rows),
            'o_comment': ['Order comment' for _ in range(n_rows)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_lineitem(self) -> pd.DataFrame:
        """Generate LINEITEM table."""
        n_rows = self.base_rows['lineitem'] * self.scale_factor
        logger.info(f"Generating LINEITEM table: {n_rows:,} rows")
        
        n_orders = self.base_rows['orders'] * self.scale_factor
        n_parts = self.base_rows['part'] * self.scale_factor
        n_suppliers = self.base_rows['supplier'] * self.scale_factor
        base_date = datetime(1992, 1, 1)
        
        data = {
            'l_orderkey': np.random.randint(1, n_orders + 1, n_rows),
            'l_partkey': np.random.randint(1, n_parts + 1, n_rows),
            'l_suppkey': np.random.randint(1, n_suppliers + 1, n_rows),
            'l_linenumber': np.random.randint(1, 8, n_rows),
            'l_quantity': np.random.randint(1, 51, n_rows),
            'l_extendedprice': np.random.uniform(900, 105000, n_rows).round(2),
            'l_discount': np.random.uniform(0, 0.10, n_rows).round(2),
            'l_tax': np.random.uniform(0, 0.08, n_rows).round(2),
            'l_returnflag': np.random.choice(['R', 'A', 'N'], n_rows),
            'l_linestatus': np.random.choice(['O', 'F'], n_rows),
            'l_shipdate': [(base_date + timedelta(days=int(np.random.randint(0, 2557)))).strftime('%Y-%m-%d') 
                          for _ in range(n_rows)],
            'l_commitdate': [(base_date + timedelta(days=int(np.random.randint(0, 2557)))).strftime('%Y-%m-%d') 
                            for _ in range(n_rows)],
            'l_receiptdate': [(base_date + timedelta(days=int(np.random.randint(0, 2557)))).strftime('%Y-%m-%d') 
                             for _ in range(n_rows)],
            'l_shipinstruct': np.random.choice(['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN'], n_rows),
            'l_shipmode': np.random.choice(['AIR', 'MAIL', 'SHIP', 'TRUCK', 'RAIL', 'REG AIR', 'FOB'], n_rows),
            'l_comment': ['Comment' for _ in range(n_rows)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_part(self) -> pd.DataFrame:
        """Generate PART table."""
        n_rows = self.base_rows['part'] * self.scale_factor
        logger.info(f"Generating PART table: {n_rows:,} rows")
        
        data = {
            'p_partkey': range(1, n_rows + 1),
            'p_name': [f'Part {i}' for i in range(1, n_rows + 1)],
            'p_mfgr': np.random.choice([f'Manufacturer#{i}' for i in range(1, 6)], n_rows),
            'p_brand': np.random.choice([f'Brand#{i}{j}' for i in range(1, 6) for j in range(1, 6)], n_rows),
            'p_type': np.random.choice(['STANDARD', 'SMALL', 'MEDIUM', 'LARGE', 'ECONOMY', 'PROMO'], n_rows),
            'p_size': np.random.randint(1, 51, n_rows),
            'p_container': np.random.choice(['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG', 'MED BAG', 'MED BOX', 'MED PKG'], n_rows),
            'p_retailprice': np.random.uniform(900, 2100, n_rows).round(2),
            'p_comment': ['Part comment' for _ in range(n_rows)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_supplier(self) -> pd.DataFrame:
        """Generate SUPPLIER table."""
        n_rows = self.base_rows['supplier'] * self.scale_factor
        logger.info(f"Generating SUPPLIER table: {n_rows:,} rows")
        
        data = {
            's_suppkey': range(1, n_rows + 1),
            's_name': [f'Supplier#{i:09d}' for i in range(1, n_rows + 1)],
            's_address': [f'Address {i}' for i in range(1, n_rows + 1)],
            's_nationkey': np.random.randint(0, 25, n_rows),
            's_phone': [f'{np.random.randint(10, 34)}-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' 
                       for _ in range(n_rows)],
            's_acctbal': np.random.uniform(-999.99, 9999.99, n_rows).round(2),
            's_comment': ['Comment' for _ in range(n_rows)]
        }
        
        return pd.DataFrame(data)
    
    def generate_all(self, output_dir: Path) -> Dict[str, Path]:
        """Generate all TPC-H tables and save to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = {}
        
        logger.info(f"Generating TPC-H SF{self.scale_factor} benchmark data")
        
        # Generate core tables
        tables = {
            'customer': self._generate_customer(),
            'orders': self._generate_orders(),
            'lineitem': self._generate_lineitem(),
            'part': self._generate_part(),
            'supplier': self._generate_supplier()
        }
        
        # Save to CSV
        for table_name, df in tables.items():
            file_path = output_dir / f'tpch_{table_name}_sf{self.scale_factor}.csv'
            df.to_csv(file_path, index=False)
            generated_files[table_name] = file_path
            logger.info(f"Saved {table_name}: {file_path} ({df.shape[0]:,} rows, {df.shape[1]} columns)")
        
        # Generate summary
        summary = {
            'scale_factor': self.scale_factor,
            'tables': {name: {'rows': df.shape[0], 'columns': df.shape[1]} for name, df in tables.items()},
            'total_rows': sum(df.shape[0] for df in tables.values()),
            'files': {name: str(path) for name, path in generated_files.items()}
        }
        
        logger.info(f"TPC-H SF{self.scale_factor} generation complete: {summary['total_rows']:,} total rows")
        return generated_files


def generate_tpch_benchmarks(data_dir: Path, config: Dict[str, Any]) -> Dict[str, Dict[str, Path]]:
    """Generate TPC-H benchmarks for all configured scale factors."""
    benchmarks = config.get('data', {}).get('benchmarks', {})
    all_generated = {}
    
    if benchmarks.get('tpch_sf1', False):
        logger.info("Generating TPC-H SF1")
        output_dir = data_dir / 'tpch_sf1'
        generator = TPCHGenerator(scale_factor=1, seed=config.get('experiment', {}).get('seed', 42))
        all_generated['tpch_sf1'] = generator.generate_all(output_dir)
    
    if benchmarks.get('tpch_sf10', False):
        logger.info("Generating TPC-H SF10")
        output_dir = data_dir / 'tpch_sf10'
        generator = TPCHGenerator(scale_factor=10, seed=config.get('experiment', {}).get('seed', 42))
        all_generated['tpch_sf10'] = generator.generate_all(output_dir)
    
    return all_generated

