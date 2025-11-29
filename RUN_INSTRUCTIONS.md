# How to Run the App

## Step 1: Open Terminal/Command Prompt

Open PowerShell, Command Prompt, or your preferred terminal.

## Step 2: Navigate to the Project Directory

```bash
cd F:\workspace\PVLDB\multi_agent_platform_evaluator
```

## Step 3: Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas duckdb faiss-cpu annoy matplotlib seaborn psutil tqdm pyyaml scikit-learn pyarrow
```

## Step 4: Run the App

```bash
python app.py
```

## Expected Output

You should see:
```
================================================================================
Research Experiment Engine - Multi-Agent Multi-Platform Evaluation
PVLDB Paper Framework
================================================================================

Step 1: Creating folder structure...
Step 2: Generating data sources...
Step 3: Initializing experiment runner...
Step 4: Running comprehensive experiments...
...
```

## Common Errors

### ModuleNotFoundError
**Solution:** Install missing dependencies:
```bash
pip install <module_name>
```

### FileNotFoundError: config/config.yaml
**Solution:** Make sure you're in the correct directory:
```bash
cd F:\workspace\PVLDB\multi_agent_platform_evaluator
```

### Import Error for src modules
**Solution:** The app should handle this automatically, but if it fails, check that all folders exist:
- `src/`
- `agents/`
- `platforms/`
- `config/`

## Alternative: Use the Batch File

You can also double-click `run_app.bat` in Windows Explorer, or run:
```bash
run_app.bat
```

## Check Results

After running, check the output in:
```
experiments/runs/<timestamp>/
```

This folder will contain:
- All metrics CSV files
- Summary tables
- Visualizations in `plots/` folder
- Analysis report: `analysis_report.md`
- Logs: `logs.txt`

