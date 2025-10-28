#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMR Regression Baseline Test Model (scikit-learn) — with plots, portable text reports, and permutation-importance

Usage:
python imr_regressor.py \
  --data path/to/data.csv \
  --target "YY_Infant_Mortality_Rate_Imr_Total_Person" \
  --id-cols State_Name State_District_Name \
  --outdir ./imr_regression_model \
  [--class-threshold 30]

Outputs (in --outdir):
  - regressor_pipeline.joblib
  - regressor_metrics.json
  - regression_report.txt
  - imr_results_summary.txt
  - classification_report.txt    # if --class-threshold flag is passed
  - feature_importance_permutation.csv
  - feature_importance_permutation.png
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
from datetime import datetime
import platform
import sys
from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.inspection import permutation_importance
import joblib


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Train an IMR regressor with metrics, plots, and portable reports.")
    p.add_argument("--data", required=True, help="Path to .csv or .parquet dataset.")
    p.add_argument("--target", required=True, help="Numeric IMR target column to predict.")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID/index columns to exclude from features.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--outdir", default="./artifacts_reg", help="Output directory.")
    p.add_argument("--class-threshold", type=float, default=None,
                   help="If set, binarize target & predictions by threshold (>= threshold -> 1) and write classification_report.txt.")
    return p.parse_args()


# Data I/O
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


# Preprocessing
def build_preprocessor_from_X(X: pd.DataFrame):
    # Detect column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Split into columns with any data vs all-missing
    num_all_missing = [c for c in num_cols if X[c].isna().all()]
    num_ok = [c for c in num_cols if c not in num_all_missing]
    cat_all_missing = [c for c in cat_cols if X[c].isna().all()]
    cat_ok = [c for c in cat_cols if c not in cat_all_missing]

    # Pipelines
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


# Plots
def save_pred_vs_actual_plot(y_true, y_pred, outpath: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
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


# Feature engineering & permutation importance
def _safe_feature_names_from_pre(preprocessor: ColumnTransformer) -> List[str]:
    """
    Best-effort extraction of output feature names from a ColumnTransformer that
    includes OneHotEncoder and scalers. Falls back to generic names on failure.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: List[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            try:
                if hasattr(trans, "named_steps"):
                    last = list(trans.named_steps.values())[-1]
                    if hasattr(last, "get_feature_names_out"):
                        fn = list(last.get_feature_names_out(cols))
                    else:
                        base_cols = cols if isinstance(cols, list) else list(cols)
                        fn = [f"{name}__{c}" for c in base_cols]
                elif hasattr(trans, "get_feature_names_out"):
                    fn = list(trans.get_feature_names_out(cols))
                else:
                    base_cols = cols if isinstance(cols, list) else list(cols)
                    fn = [f"{name}__{c}" for c in base_cols]
            except Exception:
                base_cols = cols if isinstance(cols, list) else list(cols)
                fn = [f"{name}__{c}" for c in base_cols]
            names.extend(fn)
        if not names:
            names = [f"feature_{i}" for i in range(9999)]
        return names


def compute_and_save_permutation_importance(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outdir: Path,
    random_state: int = 42,
    n_repeats: int = 5,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Runs permutation_importance on the full pipeline,
    saves bar chart PNG + CSV, and returns a DataFrame of top features.
    """
    X_imp = X_test.sample(min(1000, len(X_test)), random_state=random_state)
    y_imp = y_test.loc[X_imp.index]

    pi = permutation_importance(
        pipe, X_imp, y_imp,
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    pre = pipe.named_steps["pre"]
    feat_names_full = _safe_feature_names_from_pre(pre)

    try:
        Xt_sample = pre.transform(X_imp)
        n_out = Xt_sample.shape[1]
        if len(feat_names_full) != n_out:
            feat_names_full = [f"feature_{i}" for i in range(n_out)]
    except Exception:
        n_out = len(pi.importances_mean)
        feat_names_full = feat_names_full[:n_out] if len(feat_names_full) >= n_out else [f"feature_{i}" for i in range(n_out)]

    df_pi = pd.DataFrame({
        "feature": feat_names_full,
        "importance_mean": pi.importances_mean,
        "importance_std":  pi.importances_std,
    }).sort_values("importance_mean", ascending=False)

    csv_path = outdir / "feature_importance_permutation.csv"
    df_pi.to_csv(csv_path, index=False, encoding="utf-8")

    df_top = df_pi.head(top_k).iloc[::-1]  # reverse for better barh order
    plt.figure(figsize=(10, max(4, int(0.35 * len(df_top)))))
    plt.barh(df_top["feature"], df_top["importance_mean"], xerr=df_top["importance_std"])
    plt.xlabel("Permutation Importance (mean decrease in score)")
    plt.title(f"Top {len(df_top)} Features — Permutation Importance")
    plt.tight_layout()
    fig_path = outdir / "feature_importance_permutation.png"
    plt.savefig(fig_path)
    plt.close()

    return df_pi.head(10)


def _format_topk_for_txt(df_top10: Optional[pd.DataFrame]) -> list[str]:
    lines = ["Top 10 Features by Permutation Importance", "-" * 80]
    if df_top10 is None or df_top10.empty:
        lines.append("[unavailable]")
        return lines
    for i, row in enumerate(df_top10.itertuples(index=False), start=1):
        lines.append(f"{i:2d}. {row.feature} | mean={row.importance_mean:.6f} | std={row.importance_std:.6f}")
    return lines


# Generate reports
def write_regression_report_txt(report_path: Path, args, X, y, X_train, X_test,
                                rmse, mae, r2, evs, num_cols, cat_cols,
                                num_all_missing, cat_all_missing, pipe,
                                top10_df: Optional[pd.DataFrame] = None):
    lines = []
    lines.append("IMR Regression Report")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    lines.append("")
    lines.append("Run Configuration")
    lines.append("-" * 80)
    lines.append(f"Data: {args.data}")
    lines.append(f"Target: {args.target}")
    lines.append(f"Excluded ID cols: {', '.join(args.id_cols) if args.id_cols else '(none)'}")
    lines.append(f"Test size: {args.test_size} | Random state: {args.random_state}")
    lines.append("")
    lines.append("Dataset Shapes")
    lines.append("-" * 80)
    lines.append(f"X shape: {X.shape} | y length: {len(y)}")
    lines.append(f"Train X: {X_train.shape} | Test X: {X_test.shape}")
    lines.append("")
    lines.append("Preprocessing Summary")
    lines.append("-" * 80)
    lines.append(f"Numeric features: {len(num_cols)}")
    lines.append(f"Categorical features: {len(cat_cols)}")
    lines.append(f"All-missing numeric columns: {num_all_missing if num_all_missing else '[]'}")
    lines.append(f"All-missing categorical columns: {cat_all_missing if cat_all_missing else '[]'}")
    lines.append("")
    lines.append("Regressor")
    lines.append("-" * 80)
    lines.append(str(pipe.named_steps["reg"]))
    lines.append("")
    lines.append("Test Metrics")
    lines.append("-" * 80)
    lines.append(f"RMSE: {rmse:.6f}")
    lines.append(f"MAE: {mae:.6f}")
    lines.append(f"R^2: {r2:.6f}")
    lines.append(f"Explained Variance: {evs:.6f}")
    lines.append("")
    if top10_df is not None:
        lines.extend(_format_topk_for_txt(top10_df))
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def maybe_write_classification_report_txt(outdir: Path, threshold: float, y_true_cont, y_pred_cont):
    """
    Binarize by threshold: >= threshold -> 1 (High), else 0 (Low).
    Write classification_report.txt for portability in docs/papers.
    """
    y_true_cls = (y_true_cont >= threshold).astype(int)
    y_pred_cls = (y_pred_cont >= threshold).astype(int)

    target_names = ["Low", "High"]
    cls_rep = classification_report(y_true_cls, y_pred_cls, target_names=target_names, digits=4)
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    acc = accuracy_score(y_true_cls, y_pred_cls)

    lines = []
    lines.append("Classification-style Report (binarized from regression outputs)")
    lines.append("=" * 80)
    lines.append(f"Threshold: {threshold}  (>= threshold → class=High)")
    lines.append("")
    lines.append("Confusion Matrix [rows: true, cols: pred] (Low, High)")
    lines.append("-" * 80)
    lines.append(np.array2string(cm))
    lines.append("")
    lines.append(f"Accuracy: {acc:.6f}")
    lines.append("")
    lines.append("Detailed Classification Report")
    lines.append("-" * 80)
    lines.append(cls_rep)

    (outdir / "classification_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_portable_txt_summary(outdir: Path, args, y_test, y_pred, metrics,
                               threshold=None, top10_df: Optional[pd.DataFrame] = None):
    """
    Produce a fully portable text summary of model results for non-Python users.
    """
    lines = []
    lines.append("IMR Regression Results Summary")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 80)
    lines.append(f"Data file: {args.data}")
    lines.append(f"Target column: {args.target}")
    lines.append(f"Excluded ID cols: {', '.join(args.id_cols) if args.id_cols else '(none)'}")
    lines.append(f"Test size: {args.test_size}")
    lines.append(f"Random state: {args.random_state}")
    if threshold is not None:
        lines.append(f"Classification threshold: {threshold}")
    lines.append("")
    lines.append("Model Metrics")
    lines.append("-" * 80)
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            lines.append(f"{k.upper():25s}: {v:.6f}")
    lines.append("")
    if top10_df is not None:
        lines.extend(_format_topk_for_txt(top10_df))
        lines.append("")
        lines.append("Feature Importance Artifacts")
        lines.append("-" * 80)
        lines.append("feature_importance_permutation.csv")
        lines.append("feature_importance_permutation.png")
        lines.append("")
    lines.append("Sample Predictions (first 10)")
    lines.append("-" * 80)
    df_preview = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    lines.append(df_preview.head(10).to_string(index=False))
    lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 80)
    lines.append("regressor_pipeline.joblib")
    lines.append("regressor_metrics.json")
    lines.append("predicted_vs_actual.png")
    lines.append("residuals_vs_fitted.png")
    lines.append("residuals_hist.png")
    if threshold is not None:
        lines.append("classification_report.txt")
    lines.append("")
    lines.append("End of Summary")
    lines.append("=" * 80)

    out_txt = outdir / "imr_results_summary.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")


# Main entrypoint
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

    top10_df = None
    try:
        top10_df = compute_and_save_permutation_importance(
            pipe, X_test, y_test, outdir,
            random_state=args.random_state, n_repeats=5, top_k=20
        )
        metrics["permutation_importance_top10"] = [
            {"feature": r.feature, "importance_mean": float(r.importance_mean), "importance_std": float(r.importance_std)}
            for r in top10_df.itertuples(index=False)
        ]
    except Exception as e:
        metrics["permutation_importance_error"] = str(e)

    pipe_path = outdir / "regressor_pipeline.joblib"
    joblib.dump(pipe, pipe_path)
    with open(outdir / "regressor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_pred_vs_actual_plot(y_test.values, y_pred, outdir / "predicted_vs_actual.png")
    save_residuals_vs_fitted_plot(y_test.values, y_pred, outdir / "residuals_vs_fitted.png")
    save_residuals_hist_plot(y_test.values, y_pred, outdir / "residuals_hist.png")

    write_regression_report_txt(
        outdir / "regression_report.txt",
        args, X, y, X_train, X_test,
        rmse, mae, r2, evs, num_cols, cat_cols, num_all_missing, cat_all_missing, pipe,
        top10_df=top10_df
    )
    if args.class_threshold is not None:
        maybe_write_classification_report_txt(outdir, args.class_threshold, y_test.values, y_pred)

    write_portable_txt_summary(
        outdir, args, y_test.values, y_pred, metrics,
        threshold=args.class_threshold, top10_df=top10_df
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {pipe_path.resolve()}")
    print(f"Saved reports & plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
