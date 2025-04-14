# main.py

import sys
import os
import argparse

# Ensure project root is in Python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.core.window_runner import run_rolling_strategy

def main():
    parser = argparse.ArgumentParser(description="Run half-yearly rolling and expanding strategy")
    parser.add_argument('--stocks', nargs='+', required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--window_size_years', type=int, default=1)

    args = parser.parse_args()

    run_rolling_strategy(args)

if __name__ == "__main__":
    main()
