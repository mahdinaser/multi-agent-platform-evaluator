"""
Data generator for creating 5 types of data sources.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
from datetime import datetime, timedelta
import random
import string

logger = logging.getLogger(__name__)

class DataGenerator:
    """Generate synthetic data for experiments."""
    
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self._setup_dirs()
    
    def _setup_dirs(self):
        """Create data subdirectories."""
        self.dirs = {
            'tabular': self.data_dir / 'source_tabular',
            'logs': self.data_dir / 'source_logs',
            'vectors': self.data_dir / 'source_vectors',
            'timeseries': self.data_dir / 'source_timeseries',
            'text': self.data_dir / 'source_text'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, config: Dict):
        """Generate all data types."""
        logger.info("Generating all data sources...")
        
        # 1. Tabular data
        self.generate_tabular(config.get('tabular', {}))
        
        # 2. Log data
        self.generate_logs(config.get('logs', {}))
        
        # 3. Vector data
        self.generate_vectors(config.get('vectors', {}))
        
        # 4. Time-series data
        self.generate_timeseries(config.get('timeseries', {}))
        
        # 5. Text data
        self.generate_text(config.get('text', {}))
        
        logger.info("All data sources generated successfully!")
    
    def generate_tabular(self, config: Dict):
        """Generate tabular data with mixed numeric and categorical columns."""
        logger.info("Generating tabular data...")
        sizes = config.get('sizes', [50000, 500000, 1000000])
        max_size = config.get('max_data_size', 0)  # 0 = no limit
        if max_size > 0:
            sizes = [s for s in sizes if s <= max_size]
            if not sizes:
                logger.warning(f"All tabular sizes exceed max_data_size={max_size}, skipping")
                return
        num_categorical = config.get('num_categorical', 5)
        num_numeric = config.get('num_numeric', 10)
        
        for size in sizes:
            logger.info(f"  Generating {size:,} rows...")
            
            # Numeric columns
            data = {}
            for i in range(num_numeric):
                col_name = f'numeric_{i}'
                # Mix of distributions
                if i % 3 == 0:
                    data[col_name] = np.random.normal(100, 15, size)
                elif i % 3 == 1:
                    data[col_name] = np.random.exponential(2, size)
                else:
                    data[col_name] = np.random.uniform(0, 1000, size)
            
            # Categorical columns
            categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for i in range(num_categorical):
                col_name = f'category_{i}'
                data[col_name] = np.random.choice(categories, size)
            
            # Add ID column
            data['id'] = range(size)
            
            df = pd.DataFrame(data)
            
            # Save as parquet and CSV
            output_file = self.dirs['tabular'] / f'tabular_{size}.parquet'
            df.to_parquet(output_file, index=False)
            
            logger.info(f"  Saved to {output_file}")
    
    def generate_logs(self, config: Dict):
        """Generate log-like data with timestamps, user IDs, and event types."""
        logger.info("Generating log data...")
        num_events = config.get('num_events', 1000000)
        num_users = config.get('num_users', 10000)
        num_event_types = config.get('num_event_types', 50)
        
        # Heavy-tailed distribution for user activity
        user_weights = np.random.pareto(1.5, num_users)
        user_weights = user_weights / user_weights.sum()
        
        # Generate events
        start_time = datetime(2024, 1, 1)
        events = []
        
        for i in range(num_events):
            # Select user (heavy-tailed)
            user_id = np.random.choice(num_users, p=user_weights)
            
            # Select event type
            event_type = f'event_{np.random.randint(0, num_event_types)}'
            
            # Timestamp (some clustering)
            if np.random.random() < 0.3:
                # Clustered events
                time_offset = timedelta(seconds=np.random.exponential(60))
            else:
                time_offset = timedelta(seconds=np.random.exponential(3600))
            
            timestamp = start_time + time_offset * (i / 1000)
            
            events.append({
                'timestamp': timestamp,
                'user_id': f'user_{user_id}',
                'event_type': event_type,
                'value': np.random.exponential(10)
            })
        
        df = pd.DataFrame(events)
        output_file = self.dirs['logs'] / 'logs.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved {len(df):,} log events to {output_file}")
    
    def generate_vectors(self, config: Dict):
        """Generate vector/embedding data."""
        logger.info("Generating vector data...")
        num_vectors = config.get('num_vectors', 100000)
        dimension = config.get('dimension', 128)
        
        # Generate random vectors (normalized)
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)
        
        # Save as numpy array
        output_file = self.dirs['vectors'] / 'vectors.npy'
        np.save(output_file, vectors)
        
        # Also save metadata
        metadata = {
            'num_vectors': num_vectors,
            'dimension': dimension,
            'dtype': 'float32'
        }
        with open(self.dirs['vectors'] / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved {num_vectors:,} vectors of dimension {dimension} to {output_file}")
    
    def generate_timeseries(self, config: Dict):
        """Generate time-series data with seasonal patterns."""
        logger.info("Generating time-series data...")
        num_samples = config.get('num_samples', 1000000)
        frequency = config.get('frequency', '1min')
        
        # Create date range
        start = datetime(2024, 1, 1)
        dates = pd.date_range(start=start, periods=num_samples, freq=frequency)
        
        # Generate time series with seasonal patterns
        t = np.arange(num_samples)
        
        # Multiple seasonal components
        trend = 0.001 * t
        daily = 10 * np.sin(2 * np.pi * t / (24 * 60))  # Daily cycle (if 1min freq)
        weekly = 5 * np.sin(2 * np.pi * t / (7 * 24 * 60))
        noise = np.random.normal(0, 2, num_samples)
        
        values = trend + daily + weekly + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'category': np.random.choice(['A', 'B', 'C'], num_samples)
        })
        
        output_file = self.dirs['timeseries'] / 'timeseries.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved {len(df):,} time-series samples to {output_file}")
    
    def generate_text(self, config: Dict):
        """Generate textual data corpus."""
        logger.info("Generating text data...")
        num_documents = config.get('num_documents', 50000)
        avg_length = config.get('avg_length', 100)
        
        # Simple word pool
        words = ['data', 'analysis', 'system', 'performance', 'query', 'database',
                 'algorithm', 'optimization', 'machine', 'learning', 'research',
                 'experiment', 'evaluation', 'benchmark', 'platform', 'agent']
        
        documents = []
        for i in range(num_documents):
            # Variable length
            length = int(np.random.normal(avg_length, avg_length * 0.3))
            length = max(10, min(length, avg_length * 2))
            
            # Generate sentence
            doc_words = np.random.choice(words, size=length)
            document = ' '.join(doc_words)
            documents.append({
                'doc_id': i,
                'text': document,
                'length': len(document)
            })
        
        df = pd.DataFrame(documents)
        output_file = self.dirs['text'] / 'text_corpus.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved {len(df):,} text documents to {output_file}")

