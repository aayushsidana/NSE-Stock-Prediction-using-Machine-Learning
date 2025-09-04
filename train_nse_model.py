#!/usr/bin/env python3
"""
train_nse_model.py
Trains simple ML models on the prepared NSE feature set.
Outputs: models/, outputs/metrics.json, outputs/pred_vs_actual.png
Run: python train_nse_model.py --data data/nse_RELIANCE_NS_features.csv --target Target_Return_1d
"""
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

def train_models(df: pd.DataFrame, target_col: str):
    """
    Trains a Linear Regression and a RandomForestRegressor model.
    Splits data chronologically and handles missing values.
    
    Args:
        df: DataFrame containing features and target.
        target_col: The name of the target column.
        
    Returns:
        A tuple containing:
        - metrics (dict): MAE and R2 scores for each model.
        - y_test (np.array): The true values for the test set.
        - preds_lin (np.array): Predictions from Linear Regression.
        - preds_rf (np.array): Predictions from Random Forest.
        - scaler (object): The fitted StandardScaler object.
        - lin (object): The fitted Linear Regression model.
        - rf (object): The fitted Random Forest model.
    """
    # Exclude target and non-numeric columns from features
    exclude_cols = [target_col, "Target_Close", "Adj Close"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].values
    y = df[target_col].values

    # Remove rows with NaN or infinite values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    # Time-aware split: last 20% for testing
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train both models on the same scaled data for consistency
    lin = LinearRegression().fit(X_train_s, y_train)
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train_s, y_train)

    preds_lin = lin.predict(X_test_s)
    preds_rf = rf.predict(X_test_s)

    metrics = {
        "LinearRegression": {
            "MAE": float(mean_absolute_error(y_test, preds_lin)),
            "R2": float(r2_score(y_test, preds_lin))
        },
        "RandomForest": {
            "MAE": float(mean_absolute_error(y_test, preds_rf)),
            "R2": float(r2_score(y_test, preds_rf))
        }
    }
    return metrics, y_test, preds_lin, preds_rf, scaler, lin, rf

def plot_predictions(y_true, y_pred, outpath, title="Next-day Return: Actual vs Predicted (Test)"):
    """Plots actual vs predicted values and saves the plot."""
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time (test set index)")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="CSV from prepare_nse_data.py")
    ap.add_argument("--target", type=str, default="Target_Return_1d", help="Target column to predict")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--modeldir", type=str, default="models")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=[0], index_col=0, date_format="%Y-%m-%d")
    metrics, y_true, preds_lin, preds_rf, scaler, lin, rf = train_models(df, args.target)

    # Save models and scaler
    joblib.dump(lin, os.path.join(args.modeldir, "linear_regression_model.pkl"))
    joblib.dump(rf, os.path.join(args.modeldir, "random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(args.modeldir, "scaler.pkl"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_predictions(y_true, preds_rf, os.path.join(args.outdir, "pred_vs_actual.png"))
    
    print("Saved models, metrics, and plot.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()