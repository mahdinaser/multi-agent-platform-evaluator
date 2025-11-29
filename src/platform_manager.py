"""
Platform manager for initializing and managing backends.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PlatformManager:
    """Manages platform backends."""
    
    def __init__(self):
        self.platforms = {}
        self._initialize_platforms()
    
    def _initialize_platforms(self):
        """Initialize all available platforms."""
        platform_modules = {
            'pandas': ('platforms.pandas_backend', 'PandasBackend'),
            'duckdb': ('platforms.duckdb_backend', 'DuckDBBackend'),
            'polars': ('platforms.polars_backend', 'PolarsBackend'),
            'sqlite': ('platforms.sqlite_backend', 'SQLiteBackend'),
            'faiss': ('platforms.faiss_backend', 'FAISSBackend'),
            'annoy': ('platforms.annoy_backend', 'AnnoyBackend'),
            'baseline': ('platforms.baseline_backend', 'BaselineBackend')
        }
        
        for name, (module_path, class_name) in platform_modules.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                backend_class = getattr(module, class_name)
                self.platforms[name] = backend_class()
                logger.info(f"Initialized platform: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize platform {name}: {e}")
    
    def get_platform(self, name: str):
        """Get platform by name."""
        return self.platforms.get(name)
    
    def get_all_platforms(self) -> Dict[str, Any]:
        """Get all initialized platforms."""
        return self.platforms
    
    def get_platform_names(self) -> List[str]:
        """Get list of available platform names."""
        return list(self.platforms.keys())
    
    def is_available(self, name: str) -> bool:
        """Check if platform is available."""
        return name in self.platforms

