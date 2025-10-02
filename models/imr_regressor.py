#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMR Regression Model (scikit-learn) — with plots and robust preprocessing for all-missing columns

Usage:
python imr_regressor.py \
  --data path/to/data.csv \
  --target "YY_Infant_Mortality_Rate_Imr_Total_Person" \
  --id-cols State_Name State_District_Name \
  --outdir ./imr_regression_model

Outputs (in --outdir):
  - regressor_pipeline.joblib
  - regressor_metrics.json
  - predicted_vs_actual.png
  - residuals_vs_fitted.png
  - residuals_hist.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.inspection import permutation_importance
import joblib


def parse_args():
    p = argparse.ArgumentParser(description="Train an IMR regressor (separate script) with metrics & plots.")
    p.add_argument("--data", required=True, help="Path to .csv or .parquet dataset.")
    p.add_argument("--target", required=True, help="Numeric IMR target column to predict.")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID/index columns to exclude from features.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--outdir", default="./artifacts_reg", help="Output directory.")
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def split_Xy(df: pd.DataFrame, target: str, id_cols):
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found.")
    X = df.drop(columns=[c for c in [target] + list(id_cols) if c in df.columns])
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def build_preprocessor_from_X(X: pd.DataFrame):
    # Detect column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Split into columns with any data vs all-missing
    num_all_missing = [c for c in num_cols if X[c].isna().all()]
    num_ok = [c for c in num_cols if c not in num_all_missing]
    cat_all_missing = [c for c in cat_cols if X[c].isna().all()]
    cat_ok = [c for c in cat_cols if c not in cat_all_missing]

    # Preprocessors
    num_ok_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler())
    ])
    num_all_missing_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
        ("sc", RobustScaler())
    ])
    cat_ok_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    cat_all_missing_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_ok_pipe, num_ok),
        ("num_all_missing", num_all_missing_pipe, num_all_missing),
        ("cat", cat_ok_pipe, cat_ok),
        ("cat_all_missing", cat_all_missing_pipe, cat_all_missing),
    ])
    return pre, num_cols, cat_cols, num_all_missing, cat_all_missing


def save_pred_vs_actual_plot(y_true, y_pred, outpath: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    # Reference diagonal
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual IMR")
    plt.ylabel("Predicted IMR")
    plt.title("Predicted vs. Actual")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_residuals_vs_fitted_plot(y_true, y_pred, outpath: Path):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Fitted (Predicted) IMR")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs. Fitted")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_residuals_hist_plot(y_true, y_pred, outpath: Path):
    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=30, alpha=0.8)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df(args.data)
    X, y = split_Xy(df, args.target, args.id_cols)

    pre, num_cols, cat_cols, num_all_missing, cat_all_missing = build_preprocessor_from_X(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipe = Pipeline([
        ("pre", pre),
        ("reg", HistGradientBoostingRegressor(random_state=args.random_state))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    evs = float(explained_variance_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": evs,
        "num_features_numeric": len(num_cols),
        "num_features_categorical": len(cat_cols),
        "num_all_missing_columns": num_all_missing,
        "cat_all_missing_columns": cat_all_missing
    }

    # Permutation importance (best effort)
    try:
        X_imp = X_test.sample(min(1000, len(X_test)), random_state=args.random_state)
        y_imp = y_test.loc[X_imp.index]
        pi = permutation_importance(pipe, X_imp, y_imp, n_repeats=5, random_state=args.random_state, n_jobs=-1)
        metrics["permutation_importance"] = {
            "importances_mean": pi.importances_mean.tolist(),
            "importances_std": pi.importances_std.tolist()
        }
    except Exception as e:
        metrics["permutation_importance_error"] = str(e)

    # Save artifacts
    pipe_path = outdir / "regressor_pipeline.joblib"
    joblib.dump(pipe, pipe_path)
    with open(outdir / "regressor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    save_pred_vs_actual_plot(y_test.values, y_pred, outdir / "predicted_vs_actual.png")
    save_residuals_vs_fitted_plot(y_test.values, y_pred, outdir / "residuals_vs_fitted.png")
    save_residuals_hist_plot(y_test.values, y_pred, outdir / "residuals_hist.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {pipe_path.resolve()}")
    print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
