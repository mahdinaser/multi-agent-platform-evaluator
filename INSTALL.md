# Installation and Troubleshooting

## Common Errors and Solutions

### Error: ModuleNotFoundError

If you get import errors, install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas duckdb faiss-cpu annoy matplotlib seaborn psutil tqdm pyyaml scikit-learn pyarrow
```

### Error: Cannot find config.yaml

Make sure you're running from the correct directory:
```bash
cd PVLDB\multi_agent_platform_evaluator
python app.py
```

### Error: Import errors with src modules

The app should automatically add the project root to Python path. If you still get import errors, make sure:
1. You're in the `PVLDB\multi_agent_platform_evaluator` directory
2. All folders (`src/`, `agents/`, `platforms/`, `config/`) exist
3. All `__init__.py` files are present

### Testing Installation

Run the diagnostic script:
```bash
python diagnose.py
```

This will check all dependencies and imports.

