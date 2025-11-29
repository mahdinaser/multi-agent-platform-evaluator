"""
Utility functions for the research experiment engine.
"""
import os
import sys
import json
import yaml
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Use UTF-8 encoding to handle Unicode characters on Windows
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent

def path_join(*parts: str) -> str:
    """OS-independent path joining."""
    return str(Path(*parts))

