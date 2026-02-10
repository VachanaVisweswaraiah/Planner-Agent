#!/usr/bin/env python3
"""
Create a leaderboard by evaluating all generated output files for Calendar Scheduling.

Usage:
    python create_leaderboard.py [file1.json] [file2.json] ...
    
    If no files are provided, automatically discovers JSON files in data/ directory
    that contain calendar scheduling predictions.
"""

import json
import os
import sys
import subprocess
import re
import glob
from typing import List, Tuple, Dict
from pathlib import Path


def extract_model_info(filepath: str) -> Tuple[str, str]:
    """
    Extract model name and parameters from filepath or file contents.
    
    Returns:
        (model_name, parameters) tuple
    """
    # Try to get model from file contents
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if data:
                sample_key = list(data.keys())[0]
                sample = data[sample_key]
                if 'pred_model' in sample:
                    model_name = sample['pred_model']
                    # Extract parameters from model name if possible
                    # e.g., "google/gemma-2-9b-it" -> "gemma-2-9b"
                    if '/' in model_name:
                        model_name = model_name.split('/')[-1]
                    # Try to extract parameter count (e.g., "7b", "9b", "8b", "70b")
                    param_match = re.search(r'(\d+(?:\.\d+)?)\s*([bm])', model_name.lower())
                    if param_match:
                        params = param_match.group(1) + param_match.group(2).upper()
                    else:
                        # Try alternative patterns
                        param_match = re.search(r'(\d+)[bm]', model_name.lower())
                        params = param_match.group(0).upper() if param_match else "N/A"
                    return (model_name, params)
    except Exception as e:
        pass
    
    # Fallback: extract from filename
    filename = os.path.basename(filepath)
    # Remove common prefixes/suffixes
    name = filename.replace('output_data_calendar_', '').replace('_SLM', '').replace('calendar_scheduling', 'baseline').replace('.json', '')
    if name == '' or name == 'baseline':
        name = 'baseline'
    return (name, "N/A")


def run_evaluation(filepath: str) -> float:
    """
    Run evaluate_calendar_scheduling.py on the given file and extract solve rate.
    
    Returns:
        Solve rate as a float (0.0 to 1.0)
    """
    try:
        result = subprocess.run(
            ['python3', 'evaluate_calendar_scheduling.py', f'--data_path={filepath}'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
        )
        
        # Extract solve rate from output
        # Expected format: "Overall solve rate of 1000 samples: 0.85"
        output = result.stdout
        match = re.search(r'Overall solve rate of \d+ samples: ([\d.]+)', output)
        if match:
            return float(match.group(1))
        else:
            # Try alternative pattern
            match = re.search(r'solve rate[:\s]+([\d.]+)', output, re.IGNORECASE)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: Could not parse solve rate from output for {filepath}")
                print(f"Output: {output[:200]}")
                return 0.0
    except subprocess.TimeoutExpired:
        print(f"Error: Evaluation timed out for {filepath}")
        return 0.0
    except Exception as e:
        print(f"Error evaluating {filepath}: {e}")
        return 0.0


def discover_output_files(data_dir: str = "data") -> List[str]:
    """
    Automatically discover JSON files that contain calendar scheduling predictions.
    
    Returns:
        List of file paths
    """
    files = []
    
    # Check data directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                continue
                
            # Check if it's a calendar scheduling file with predictions
            sample_key = list(data.keys())[0]
            sample = data[sample_key]
            
            # Must have both pred_5shot_pro and golden_plan
            if 'pred_5shot_pro' in sample and 'golden_plan' in sample:
                # Check if it looks like calendar scheduling (has num_people, num_days)
                if 'num_people' in sample and 'num_days' in sample:
                    files.append(json_file)
        except Exception as e:
            continue
    
    return files


def create_leaderboard(filepaths: List[str]) -> List[Tuple[str, str, str, float]]:
    """
    Evaluate all files and create leaderboard entries.
    
    Returns:
        List of (rank, model, parameters, solve_rate) tuples, sorted by solve_rate descending
    """
    results = []
    
    print("=" * 70)
    print("EVALUATING FILES")
    print("=" * 70)
    print()
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"Evaluating: {filepath}")
        model_name, parameters = extract_model_info(filepath)
        solve_rate = run_evaluation(filepath)
        
        results.append((model_name, parameters, solve_rate, filepath))
        print(f"  Model: {model_name}")
        print(f"  Parameters: {parameters}")
        print(f"  Solve Rate: {solve_rate:.4f}")
        print()
    
    # Sort by solve rate (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Add ranks
    leaderboard = []
    for rank, (model, params, solve_rate, _) in enumerate(results, start=1):
        leaderboard.append((str(rank), model, params, solve_rate))
    
    return leaderboard


def print_leaderboard(leaderboard: List[Tuple[str, str, str, float]]):
    """
    Print leaderboard in the requested format.
    """
    print("=" * 70)
    print("LEADERBOARD - Calendar Scheduling")
    print("=" * 70)
    print()
    # Print header exactly as requested
    print("| Rank | Model | Parameters | Solve Rate |")
    print("|------|-------|------------|------------|")
    
    for rank, model, params, solve_rate in leaderboard:
        # Format solve rate with 4 decimal places
        solve_rate_str = f"{solve_rate:.4f}"
        print(f"| {rank} | {model} | {params} | {solve_rate_str} |")
    
    print()


def main():
    # Get file paths from command line or auto-discover
    if len(sys.argv) > 1:
        filepaths = sys.argv[1:]
    else:
        print("No files specified. Auto-discovering output files...")
        filepaths = discover_output_files()
        
        if not filepaths:
            print("No output files found with predictions.")
            print("Usage: python create_leaderboard.py [file1.json] [file2.json] ...")
            sys.exit(1)
        
        print(f"Found {len(filepaths)} file(s) with predictions:")
        for fp in filepaths:
            print(f"  - {fp}")
        print()
    
    # Create leaderboard
    leaderboard = create_leaderboard(filepaths)
    
    # Print results
    print_leaderboard(leaderboard)


if __name__ == "__main__":
    main()
