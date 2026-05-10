#!/usr/bin/env python3
"""
Dynamic Runtime Detection Harness

This script runs each binary in the dataset and captures dynamic runtime features using:
1. `perf stat` (hardware performance counters: instructions, cycles, branches, branch-misses)
2. `strace -c` (system call summaries: total syscalls, getrandom counts)

It outputs a new CSV file: `../dataset/dynamic_features.csv`.
"""

import os
import json
import subprocess
import pandas as pd
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
BIN_DIR = os.path.join(DATA_DIR, "binaries")
METADATA_FILE = os.path.join(DATA_DIR, "binary_metadata.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "dynamic_features.csv")

def run_perf(binary_path):
    """Run binary with perf stat and capture metrics."""
    # We use -x, to get CSV-like output from perf
    cmd = [
        "perf", "stat", "-x,", 
        "-e", "instructions,cycles,branches,branch-misses",
        binary_path
    ]
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        perf_output = result.stderr
        
        metrics = {
            "dyn_instructions": 0.0,
            "dyn_cycles": 0.0,
            "dyn_branches": 0.0,
            "dyn_branch_misses": 0.0
        }
        
        for line in perf_output.splitlines():
            parts = line.split(',')
            if len(parts) >= 3:
                val_str = parts[0].strip()
                event = parts[2].strip()
                
                if not val_str or val_str == '<not counted>':
                    continue
                    
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                    
                if event == 'instructions':
                    metrics['dyn_instructions'] = val
                elif event == 'cycles':
                    metrics['dyn_cycles'] = val
                elif event == 'branches':
                    metrics['dyn_branches'] = val
                elif event == 'branch-misses':
                    metrics['dyn_branch_misses'] = val
                    
        return metrics
    except subprocess.TimeoutExpired:
        print(f"  [Timeout] perf {binary_path}")
        return None
    except Exception as e:
        print(f"  [Error] perf {binary_path}: {e}")
        return None

def run_strace(binary_path):
    """Run binary with strace -c and capture syscall metrics."""
    cmd = ["strace", "-c", binary_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        strace_output = result.stderr
        
        metrics = {
            "dyn_total_syscalls": 0.0,
            "dyn_unique_syscalls": 0.0,
            "dyn_getrandom_calls": 0.0,
            "dyn_read_calls": 0.0,
            "dyn_write_calls": 0.0
        }
        
        lines = strace_output.splitlines()
        parsing = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("------"):
                parsing = not parsing
                continue
                
            if parsing:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        calls = float(parts[3])
                        syscall = parts[-1]
                        
                        metrics['dyn_total_syscalls'] += calls
                        metrics['dyn_unique_syscalls'] += 1
                        
                        if syscall == 'getrandom':
                            metrics['dyn_getrandom_calls'] += calls
                        elif syscall == 'read':
                            metrics['dyn_read_calls'] += calls
                        elif syscall == 'write':
                            metrics['dyn_write_calls'] += calls
                    except ValueError:
                        pass
        return metrics
    except subprocess.TimeoutExpired:
        print(f"  [Timeout] strace {binary_path}")
        return None
    except Exception as e:
        print(f"  [Error] strace {binary_path}: {e}")
        return None

def analyze_binary(meta):
    """Analyze a single binary and return its dynamic features."""
    binary_path = os.path.join(BIN_DIR, meta['binary_name'])
    
    if not os.path.exists(binary_path):
        return None
        
    start_time = time.time()
    try:
        # Just run it once normally to get execution time
        subprocess.run([binary_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except:
        pass
    exec_time = time.time() - start_time
    
    perf_metrics = run_perf(binary_path)
    strace_metrics = run_strace(binary_path)
    
    if not perf_metrics or not strace_metrics:
        return None
        
    features = {
        "binary_name": meta['binary_name'],
        "label": meta['label'],
        "dyn_exec_time": exec_time,
    }
    
    features.update(perf_metrics)
    features.update(strace_metrics)
    
    # Derived metrics
    features['dyn_ipc'] = features['dyn_instructions'] / max(features['dyn_cycles'], 1)
    features['dyn_branch_miss_ratio'] = features['dyn_branch_misses'] / max(features['dyn_branches'], 1)
    
    return features

def main():
    print("=" * 70)
    print("DYNAMIC RUNTIME FEATURE EXTRACTION")
    print("=" * 70)
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return
        
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
        
    print(f"Found {len(metadata)} binaries to analyze dynamically.")
    
    results = []
    
    # We use ThreadPoolExecutor to speed up the process, but not too many threads
    # to avoid distorting performance metrics.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_binary, meta): meta for meta in metadata}
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(metadata)} binaries...")
                
    if not results:
        print("Error: No dynamic features were extracted. Check if perf/strace are working.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nFeature extraction complete:")
    print(f"  Successfully processed: {len(df)}")
    print(f"  Failed: {len(metadata) - len(df)}")
    print(f"  Output saved to: {OUTPUT_FILE}")
    
    print("\nSample Dynamic Features:")
    print(df[['binary_name', 'dyn_instructions', 'dyn_ipc', 'dyn_branch_miss_ratio', 'dyn_total_syscalls']].head())

if __name__ == "__main__":
    main()
