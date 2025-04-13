#!/usr/bin/env python3

import os
import sys
import argparse
import time
import subprocess
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    """Print a formatted header for different sections"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def run_command(command, description=None):
    """Run a shell command and print status"""
    if description:
        print(f"\n{description}...\n")
    
    print(f"Running: {command}")
    start_time = time.time()
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"--- Command completed successfully in {elapsed_time:.2f} seconds")
        # Print output
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        return True
    else:
        print(f"--- Command failed with return code {result.returncode} after {elapsed_time:.2f} seconds")
        if result.stderr:
            print("\nError:")
            print(result.stderr)
        return False

def run_python_script(script_name, args="", description=None):
    """Run a Python script with given arguments"""
    # Get the path to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"--- Script not found: {script_path}")
        return False
    
    # Run the script using sys.executable to ensure it uses the same Python interpreter
    # and properly quote the path to handle spaces
    command = f'"{sys.executable}" "{script_path}" {args}'
    return run_command(command, description)

def create_results_folder():
    """Create a folder for results without timestamp"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def main():
    """Main function to run the pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the stock market prediction pipeline.")
    
    # General options
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "tune", "evaluate", "all"],
                        help="Pipeline mode (train, tune, evaluate, all)")
    
    # Model options
    parser.add_argument("--model", type=str, default="logistic", 
                        choices=["logistic", "rf", "xgb", "lstm", "attention", "all"],
                        help="Model to use or tune")
    
    # Data options
    parser.add_argument("--stocks", type=str, nargs="+", 
                        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help="Stock symbols to use")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Start date for data filtering (format: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None,
                        help="End date for data filtering (format: YYYY-MM-DD)")
    parser.add_argument("--year", type=int, default=None,
                        help="Specific year to use for data (e.g., 2016)")
    
    # Hyperparameter tuning options
    parser.add_argument("--tuning_method", type=str, default="random",
                        choices=["grid", "random"],
                        help="Method for hyperparameter tuning")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="Number of iterations for randomized search")
    
    args = parser.parse_args()
    
    print_header("STOCK MARKET PREDICTION PIPELINE")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Stocks: {args.stocks}")
    print(f"Test Size: {args.test_size}")
    
    if args.start_date and args.end_date:
        print(f"Date Range: {args.start_date} to {args.end_date}")
    elif args.year:
        print(f"Year: {args.year}")
    
    # Create timestamped results folder
    results_dir = create_results_folder()
    print(f"\n--- Results will be saved to: {results_dir}")
    
    # Build common arguments
    common_args = f"--model {args.model} --test_size {args.test_size}"
    
    # Add stock arguments
    if len(args.stocks) > 0:
        common_args += " --stocks"
        for stock in args.stocks:
            common_args += f" {stock}"
    
    # Add date filtering arguments
    if args.year:
        common_args += f" --year {args.year}"
    else:
        if args.start_date:
            common_args += f" --start_date {args.start_date}"
        if args.end_date:
            common_args += f" --end_date {args.end_date}"
    
    # Run pipeline based on mode
    if args.mode == "tune" or args.mode == "all":
        print_header("HYPERPARAMETER TUNING")
        
        # Build command for hyperparameter tuning
        hyperparams_args = common_args
        
        if args.tuning_method != "random":
            hyperparams_args += f" --tuning_method {args.tuning_method}"
        if args.n_iter != 20:
            hyperparams_args += f" --n_iter {args.n_iter}"
        
        # Run hyperparameter tuning
        success = run_python_script(
            "hyperparams.py", 
            hyperparams_args,
            f"Tuning hyperparameters for {args.model} model"
        )
        
        if not success and args.mode == "all":
            print("--- Hyperparameter tuning failed. Continuing with default parameters.")
    
    if args.mode == "train" or args.mode == "all":
        print_header("MODEL TRAINING")
        
        # Run training
        success = run_python_script(
            "train.py", 
            common_args,
            f"Training {args.model} model"
        )
        
        if not success and args.mode == "all":
            print("--- Training failed. Stopping pipeline.")
            return
    
    if args.mode == "evaluate" or args.mode == "all":
        print_header("MODEL EVALUATION")
        
        # Run evaluation
        success = run_python_script(
            "evaluate.py", 
            common_args,
            f"Evaluating {args.model} model"
        )
        
        if not success:
            print("--- Evaluation failed.")
    
    print_header("PIPELINE COMPLETED")

if __name__ == "__main__":
    main()