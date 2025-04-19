import sys
import os
import argparse

# Optionally, reduce TensorFlow logging verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all logs; 1 = INFO messages are suppressed; 2 = WARNINGS are suppressed; 3 = ERROR only

# Ensure project root is added to the Python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.core.window_runner import run_rolling_strategy

def beep():
    try:
        if sys.platform == "win32":
            # On Windows, use winsound module to beep at 1000 Hz for 500 ms
            import winsound
            winsound.Beep(1000, 500)
        else:
            # On other platforms, printing ASCII Bell ('\a') may trigger a sound if the terminal is configured to do so.
            print('\a', end='', flush=True)
    except Exception as e:
        print("Beep failed:", e)

def main():
    parser = argparse.ArgumentParser(description="Run half-yearly rolling and expanding strategy")
    parser.add_argument('--stocks', nargs='+', required=True, help="List of stock tickers, e.g., AAPL GOOGL")
    parser.add_argument('--model', type=str, nargs='+', 
                    choices=['logistic', 'rf', 'xgb', 'lstm', 'svm', 'all'],
                    help='Model type(s) to train')
    parser.add_argument('--start_date', type=str, required=True, help="Start date for data window (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, required=True, help="End date for data window (YYYY-MM-DD)")
    parser.add_argument('--window_size_years', type=int, default=1, help="Window size in years (default: 1)")

    args = parser.parse_args()
    run_rolling_strategy(args)
    
    # Trigger sound when processing completes.
    beep()

if __name__ == "__main__":
    main()
