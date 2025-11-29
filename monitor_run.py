#!/usr/bin/env python3
"""Monitor the latest experiment run for Ollama usage and errors."""
import os
from pathlib import Path
import time

project_root = Path(__file__).parent
runs_dir = project_root / 'experiments' / 'runs'

print("Monitoring experiment runs...")
print("=" * 70)

last_run = None
while True:
    # Find latest run
    if runs_dir.exists():
        runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            current_run = runs[0]
            if current_run != last_run:
                last_run = current_run
                print(f"\n📁 New run detected: {current_run.name}")
            
            log_file = current_run / 'logs.txt'
            if log_file.exists():
                # Read last 30 lines
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    recent = lines[-30:] if len(lines) > 30 else lines
                    
                    # Check for key indicators
                    ollama_found = any('Ollama initialized' in line or '✓ Ollama' in line for line in recent)
                    llm_agent_found = any('Initialized agent: llm' in line for line in recent)
                    errors = [line for line in recent if 'ERROR' in line or 'WARNING' in line and 'Ollama' in line]
                    
                    if ollama_found:
                        print("✅ Ollama is being used!")
                    elif llm_agent_found:
                        print("ℹ️  LLM agent initialized (checking Ollama status...)")
                    
                    if errors:
                        print("\n⚠️  Recent warnings/errors:")
                        for err in errors[-5:]:
                            print(f"   {err.strip()}")
                    
                    # Show last few lines
                    print("\n📝 Latest log entries:")
                    for line in recent[-5:]:
                        print(f"   {line.rstrip()}")
    
    time.sleep(5)

