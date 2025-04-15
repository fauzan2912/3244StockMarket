# src/utils/io.py

import os
import pickle
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def save_model(model, model_type, stock_symbol, suffix="latest"):
    path = os.path.join(RESULTS_DIR, stock_symbol)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"model_{suffix}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[✓] Model saved: {filepath}")

def save_feature_cols(feature_cols, stock_symbol, model_type, suffix="latest"):
    path = os.path.join(RESULTS_DIR, stock_symbol)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"features_{suffix}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"[✓] Feature columns saved: {filepath}")

def save_metrics(stock_symbol, metrics, suffix="latest"):
    path = os.path.join(RESULTS_DIR, stock_symbol)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"metrics_{suffix}.json")
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[✓] Metrics saved: {filepath}")

def load_model(model_type, stock_symbol, suffix="latest"):
    filepath = os.path.join(RESULTS_DIR, stock_symbol, f"model_{suffix}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_feature_cols(stock_symbol, model_type, suffix="latest"):
    filepath = os.path.join(RESULTS_DIR, stock_symbol, f"features_{suffix}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature columns not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
