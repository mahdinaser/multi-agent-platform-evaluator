"""
NYC Taxi benchmark data loader/generator.
Either downloads real NYC Taxi data or generates synthetic taxi-like data.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NYCTaxiLoader:
    """Loads or generates NYC Taxi benchmark data."""
    
    def __init__(self, num_rows: int = 1_000_000, seed: int = 42, use_real_data: bool = False):
        """
        Initialize NYC Taxi loader.
        
        Args:
            num_rows: Number of rows to generate (if not using real data)
            seed: Random seed for reproducibility
            use_real_data: If True, attempt to download real NYC Taxi data
        """
        self.num_rows = num_rows
        self.seed = seed
        self.use_real_data = use_real_data
        np.random.seed(seed)
    
    def _download_real_data(self, output_dir: Path) -> Optional[pd.DataFrame]:
        """
        Attempt to download real NYC Taxi data.
        Falls back to synthetic if download fails.
        """
        try:
            # NYC Taxi data URLs (example for 2023-01)
            url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
            logger.info(f"Attempting to download real NYC Taxi data from {url}")
            
            df = pd.read_parquet(url)
            logger.info(f"Successfully downloaded {len(df):,} taxi trip records")
            
            # Sample if too large
            if len(df) > self.num_rows:
                df = df.sample(n=self.num_rows, random_state=self.seed)
                logger.info(f"Sampled {self.num_rows:,} records")
            
            return df
        
        except Exception as e:
            logger.warning(f"Failed to download real NYC Taxi data: {e}")
            logger.info("Falling back to synthetic taxi data generation")
            return None
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic NYC Taxi-like data."""
        logger.info(f"Generating synthetic NYC Taxi data: {self.num_rows:,} rows")
        
        base_date = datetime(2023, 1, 1)
        
        # Generate pickup times (spread across January 2023)
        pickup_times = [base_date + timedelta(seconds=np.random.randint(0, 31*24*3600)) 
                       for _ in range(self.num_rows)]
        
        # Trip duration in seconds (typically 5-60 minutes)
        trip_durations = np.random.gamma(shape=2, scale=600, size=self.num_rows).astype(int)
        trip_durations = np.clip(trip_durations, 60, 7200)  # 1 min to 2 hours
        
        # Dropoff times
        dropoff_times = [pickup_times[i] + timedelta(seconds=int(trip_durations[i])) 
                        for i in range(self.num_rows)]
        
        # Trip distance in miles (typically 1-10 miles)
        trip_distances = np.random.gamma(shape=2, scale=2.5, size=self.num_rows)
        trip_distances = np.clip(trip_distances, 0.1, 50.0).round(2)
        
        # Fare amount (base fare $2.50 + $2.50/mile + time)
        base_fare = 2.50
        per_mile = 2.50
        fare_amounts = (base_fare + per_mile * trip_distances + 
                       trip_durations / 60 * 0.50)  # $0.50 per minute
        fare_amounts = fare_amounts.round(2)
        
        # Additional charges
        extra_charges = np.random.choice([0.0, 0.50, 1.0], self.num_rows, p=[0.7, 0.2, 0.1])
        mta_tax = 0.50
        tip_amounts = (fare_amounts * np.random.uniform(0, 0.25, self.num_rows)).round(2)
        tolls_amounts = np.random.choice([0.0, 2.75, 5.76, 6.50], self.num_rows, p=[0.85, 0.05, 0.05, 0.05])
        
        total_amounts = (fare_amounts + extra_charges + mta_tax + tip_amounts + tolls_amounts).round(2)
        
        # Location coordinates (Manhattan bounding box approximately)
        pickup_longitude = np.random.uniform(-74.02, -73.93, self.num_rows).round(6)
        pickup_latitude = np.random.uniform(40.70, 40.80, self.num_rows).round(6)
        dropoff_longitude = np.random.uniform(-74.02, -73.93, self.num_rows).round(6)
        dropoff_latitude = np.random.uniform(40.70, 40.80, self.num_rows).round(6)
        
        # Categorical variables
        vendor_ids = np.random.choice([1, 2], self.num_rows, p=[0.6, 0.4])
        passenger_counts = np.random.choice([1, 2, 3, 4, 5, 6], self.num_rows, 
                                          p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
        rate_codes = np.random.choice([1, 2, 3, 4, 5, 6], self.num_rows, 
                                     p=[0.9, 0.04, 0.02, 0.02, 0.01, 0.01])
        payment_types = np.random.choice([1, 2, 3, 4], self.num_rows, 
                                        p=[0.7, 0.25, 0.03, 0.02])
        
        data = {
            'VendorID': vendor_ids,
            'tpep_pickup_datetime': pickup_times,
            'tpep_dropoff_datetime': dropoff_times,
            'passenger_count': passenger_counts,
            'trip_distance': trip_distances,
            'pickup_longitude': pickup_longitude,
            'pickup_latitude': pickup_latitude,
            'RatecodeID': rate_codes,
            'dropoff_longitude': dropoff_longitude,
            'dropoff_latitude': dropoff_latitude,
            'payment_type': payment_types,
            'fare_amount': fare_amounts,
            'extra': extra_charges,
            'mta_tax': mta_tax,
            'tip_amount': tip_amounts,
            'tolls_amount': tolls_amounts,
            'total_amount': total_amounts,
            'trip_duration_seconds': trip_durations
        }
        
        df = pd.DataFrame(data)
        
        # Sort by pickup time
        df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
        
        return df
    
    def generate(self, output_dir: Path) -> Path:
        """Generate or load NYC Taxi data and save to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try real data first if requested
        if self.use_real_data:
            df = self._download_real_data(output_dir)
        else:
            df = None
        
        # Fall back to synthetic if real data not available
        if df is None:
            df = self._generate_synthetic_data()
        
        # Save to CSV
        file_path = output_dir / 'nyc_taxi_trips.csv'
        df.to_csv(file_path, index=False)
        
        logger.info(f"Saved NYC Taxi data: {file_path} ({df.shape[0]:,} rows, {df.shape[1]} columns)")
        logger.info(f"Date range: {df['tpep_pickup_datetime'].min()} to {df['tpep_pickup_datetime'].max()}")
        logger.info(f"Total fare: ${df['total_amount'].sum():,.2f}, Avg trip: {df['trip_distance'].mean():.2f} miles")
        
        return file_path


def generate_nyc_taxi_benchmark(data_dir: Path, config: Dict[str, Any]) -> Optional[Path]:
    """Generate NYC Taxi benchmark if configured."""
    benchmarks = config.get('data', {}).get('benchmarks', {})
    
    if benchmarks.get('nyc_taxi', False):
        logger.info("Generating NYC Taxi benchmark")
        output_dir = data_dir / 'nyc_taxi'
        
        # Determine number of rows
        num_rows = config.get('data', {}).get('nyc_taxi_rows', 1_000_000)
        use_real = config.get('data', {}).get('nyc_taxi_use_real', False)
        
        loader = NYCTaxiLoader(
            num_rows=num_rows,
            seed=config.get('experiment', {}).get('seed', 42),
            use_real_data=use_real
        )
        
        file_path = loader.generate(output_dir)
        return file_path
    
    return None

