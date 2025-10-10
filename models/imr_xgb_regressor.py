#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMR XGBoost Regression Model — XGBoost 3.x optimized

Usage:
python imr_xgb_regressor.py \
  --data path/to/data.csv \
  --target "YY_Infant_Mortality_Rate_Imr_Total_Person" \
  --id-cols State_Name State_District_Name \
  --outdir ./imr_xgb_regression_model \
  [--class-threshold 30] \
  [--eval-size 0.15 --early-stopping-rounds 100] \
  [--gpu] \
  [--n-estimators 2000 --learning-rate 0.03 --max-depth 6 --subsample 0.8 --colsample-bytree 0.8] \
  [--progress]

Outputs (in --outdir):
  - regressor_pipeline.joblib
  - regressor_metrics.json
  - regression_report.txt
  - imr_results_summary.txt
  - classification_report.txt
  - feature_importance_permutation.csv
  - feature_importance_permutation.png
  - predicted_vs_actual.png
  - residuals_vs_fitted.png
  - residuals_hist.png
"""

from __future__ import annotations
import argparse, json
import inspect
import math
from pathlib import Path
from datetime import datetime
import platform, sys
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.inspection import permutation_importance
import joblib
from xgboost import XGBRegressor, __version__ as xgb_version
from xgboost.callback import EarlyStopping as XEarlyStopping, TrainingCallback

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Train an IMR XGBoost regressor with metrics, plots, and portable reports.")
    p.add_argument("--data", required=True, help="Path to .csv or .parquet dataset.")
    p.add_argument("--target", required=True, help="Numeric IMR target column to predict.")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID/index columns to exclude from features.")
    p.add_argument("--outdir", default="./artifacts_reg_xgb", help="Output directory.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--eval-size", type=float, default=0.15, help="Validation fraction carved from TRAIN for early stopping (0 disables).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=1.5)
    p.add_argument("--reg-alpha", type=float, default=0.0)     # L1
    p.add_argument("--reg-lambda", type=float, default=1.0)    # L2
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--max-bin", type=int, default=256, help="Max histogram bins (tree_method=hist).")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--early-stopping-rounds", type=int, default=100, help="0 disables early stopping.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available (device='cuda').")
    p.add_argument("--verbosity", type=int, default=0, help="XGBoost fit verbosity (0=quiet).")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars during training and steps.")
    p.add_argument("--class-threshold", type=float, default=None,
                   help="If set, binarize target & predictions by threshold (>= threshold -> 1) and write classification_report.txt.")
    return p.parse_args()


def _null_step():
    class Dummy:
        def update(self, *_a, **_k): pass
        def close(self): pass
    return Dummy()

def step_ctx(msg: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(total=1, desc=msg, leave=False)
    return _null_step()


# Data I/O
def load_df(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {suffix}")

def split_Xy(df: pd.DataFrame, target: str, id_cols) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found.")
    X = df.drop(columns=[c for c in [target] + list(id_cols) if c in df.columns])
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def build_preprocessor_from_X(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_all_missing = [c for c in num_cols if X[c].isna().all()]
    num_ok = [c for c in num_cols if c not in num_all_missing]
    cat_all_missing = [c for c in cat_cols if X[c].isna().all()]
    cat_ok = [c for c in cat_cols if c not in cat_all_missing]

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

def _adjusted_r2(r2: float, n: int, p: int) -> float:
    denom = max(n - p - 1, 1)
    return 1.0 - (1.0 - r2) * (n - 1) / denom

def _effective_num_features_from_pre(pre, X_sample) -> int:
    try:
        return int(pre.transform(X_sample[:5]).shape[1])
    except Exception:
        try:
            return int(len(pre.get_feature_names_out()))
        except Exception:
            return int(X_sample.shape[1])

def save_baseline_scatter_plot(
    y_true, y_pred, title: str, outpath: Path,
    target_label: str = "Infant Mortality Rate",
    n_features: int | None = None
):
    order = np.argsort(np.asarray(y_true))
    y_true_sorted = np.asarray(y_true)[order]
    y_pred_sorted = np.asarray(y_pred)[order]

    rmse = float(np.sqrt(mean_squared_error(y_true_sorted, y_pred_sorted)))
    r2 = float(r2_score(y_true_sorted, y_pred_sorted))
    n = len(y_true_sorted)
    p = int(n_features) if n_features is not None else 1
    adj_r2 = float(_adjusted_r2(r2, n=n, p=p))

    plt.figure(figsize=(8, 4.2))
    x = np.arange(n)
    plt.plot(x, y_pred_sorted, linewidth=1.5, label="Predicted Value")   # blue line
    plt.scatter(x, y_true_sorted, s=22, color="k", label="True Value")   # black dots

    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(target_label)
    plt.legend(loc="upper left", frameon=True)

    metrics_text = f"Metrics:\nR^2: {r2:.2f}\nAdjusted R^2: {adj_r2:.2f}\nRMSE: {rmse:.2f}"
    plt.text(
        1.02, 0.98, metrics_text,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7")
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


# Features and permutation importance
def _safe_feature_names_from_pre(preprocessor: ColumnTransformer) -> List[str]:
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

    (outdir / "feature_importance_permutation.csv").write_text(
        df_pi.to_csv(index=False, encoding="utf-8")
    )

    df_top = df_pi.head(top_k).iloc[::-1]
    plt.figure(figsize=(10, max(4, int(0.35 * len(df_top)))))
    plt.barh(df_top["feature"], df_top["importance_mean"], xerr=df_top["importance_std"])
    plt.xlabel("Permutation Importance (mean decrease in score)")
    plt.title("Top {} Features — Permutation Importance (XGB)".format(len(df_top)))
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance_permutation.png")
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
    lines.append("IMR XGBoost Regression Report")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")
    lines.append("")
    lines.append("Run Configuration")
    lines.append("-" * 80)
    lines.append(f"Data: {args.data}")
    lines.append(f"Target: {args.target}")
    lines.append(f"Excluded ID cols: {', '.join(args.id_cols) if args.id_cols else '(none)'}")
    lines.append(f"Test size: {args.test_size} | Eval size: {args.eval_size} | Random state: {args.random_state}")
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
    lines = []
    lines.append("IMR XGBoost Regression Results Summary")
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
    lines.append(f"Eval size: {args.eval_size}")
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

    (outdir / "imr_results_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# Model runs and progress
def _fit_accepts_callbacks(model) -> bool:
    """Return True if model.fit(...) supports the 'callbacks' kwarg."""
    try:
        sig = inspect.signature(model.fit)
        return "callbacks" in sig.parameters
    except Exception:
        return False

def _fit_accepts(model, param: str) -> bool:
    try:
        return param in inspect.signature(model.fit).parameters
    except Exception:
        return False

class TQDMCallback(TrainingCallback):
    """tqdm progress across boosting rounds (XGBoost 3.x)."""
    def __init__(self, total_rounds: int, enable: bool):
        self.total_rounds = int(total_rounds)
        self.enable = bool(enable and (tqdm is not None))
        self._pbar = None
    def before_training(self, model):
        if self.enable:
            self._pbar = tqdm(total=self.total_rounds, desc="XGBoost boosting rounds", leave=False)
        return model
    def after_iteration(self, model, epoch: int, evals_log):
        if self._pbar:
            self._pbar.update(1)
        return False
    def after_training(self, model):
        if self._pbar:
            self._pbar.close()
        return model


# Main entrypoint
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def step(msg: str):
        return step_ctx(msg, enabled=args.progress)

    with step("Load data"):
        df = load_df(args.data)
        X, y = split_Xy(df, args.target, args.id_cols)

    with step("Build preprocessor"):
        pre, num_cols, cat_cols, num_all_missing, cat_all_missing = build_preprocessor_from_X(X)

    with step("Train/test split"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

    X_tr, X_ev, y_tr, y_ev = X_train, None, y_train, None
    use_early_stop = (args.early_stopping_rounds and args.early_stopping_rounds > 0
                      and args.eval_size and args.eval_size > 0.0)
    if use_early_stop:
        with step("Make eval split"):
            X_tr, X_ev, y_tr, y_ev = train_test_split(
                X_train, y_train,
                test_size=args.eval_size,
                random_state=args.random_state
            )

    xgb = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        max_bin=args.max_bin,
        objective="reg:squarederror",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbosity=args.verbosity,
        eval_metric="rmse",
        tree_method="hist",
        device=("cuda" if args.gpu else "cpu"),
    )

    pipe = Pipeline([("pre", pre), ("reg", xgb)])

    if use_early_stop:
        with step("Fit preprocessor"):
            pre.fit(X_tr, y_tr)
        with step("Transform train/eval"):
            Xt_tr = pre.transform(X_tr)
            Xt_ev = pre.transform(X_ev)

        # Choose early-stopping API based on what's actually supported
        supports_callbacks = _fit_accepts_callbacks(xgb)
        supports_es_rounds = _fit_accepts(xgb, "early_stopping_rounds")

        if supports_callbacks:
            cb = [XEarlyStopping(rounds=args.early_stopping_rounds, save_best=True)]
            if args.progress and tqdm is not None:
                cb.append(TQDMCallback(total_rounds=args.n_estimators, enable=True))
            fit_kwargs = {"eval_set": [(Xt_ev, y_ev)], "callbacks": cb}
            if _fit_accepts(xgb, "verbose"):
                fit_kwargs["verbose"] = (args.verbosity > 0)
            elif _fit_accepts(xgb, "verbose_eval"):
                fit_kwargs["verbose_eval"] = (args.verbosity > 0)
            xgb.fit(Xt_tr, y_tr, **fit_kwargs)

        elif supports_es_rounds:
            fit_kwargs = {"eval_set": [(Xt_ev, y_ev)], "early_stopping_rounds": args.early_stopping_rounds}
            if _fit_accepts(xgb, "verbose"):
                fit_kwargs["verbose"] = (args.verbosity > 0)
            elif _fit_accepts(xgb, "verbose_eval"):
                fit_kwargs["verbose_eval"] = (args.verbosity > 0)
            xgb.fit(Xt_tr, y_tr, **fit_kwargs)

        else:
            # Neither API is available; train without ES
            if args.progress and tqdm is not None:
                _ = TQDMCallback(total_rounds=args.n_estimators, enable=False)  # keeps behavior consistent
            pipe = Pipeline([("pre", pre), ("reg", xgb)])
            pipe.fit(X_train, y_train)

        pipe = Pipeline([("pre", pre), ("reg", xgb)])
    else:
        with step("Fit pipeline"):
            pipe.fit(X_train, y_train)

    with step("Predict"):
        y_pred = pipe.predict(X_test)

    y_pred = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    pre_fitted = pipe.named_steps["pre"]
    n_features_eff = _effective_num_features_from_pre(pre_fitted, X_train)

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
        "cat_all_missing_columns": cat_all_missing,
        "best_iteration": getattr(xgb, "best_iteration", None),
        "xgboost_version": xgb_version,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
        "max_bin": args.max_bin,
        "device": "cuda" if args.gpu else "cpu",
        "tree_method": "hist",
        "early_stopping_rounds": args.early_stopping_rounds,
        "eval_size": args.eval_size if use_early_stop else 0.0,
        "fit_supports_callbacks": supports_callbacks if use_early_stop else None,
        "fit_supports_es_rounds": supports_es_rounds if use_early_stop else None,
    }

    save_baseline_scatter_plot(
        y_true=y_train, y_pred=y_pred_train,
        title="Baseline Training Scatter Plot",
        outpath=outdir / "baseline_training_scatter.png",
        target_label="Infant Mortality Rate",
        n_features=n_features_eff
    )
    save_baseline_scatter_plot(
        y_true=y_test, y_pred=y_pred,
        title="Baseline Testing Scatter Plot",
        outpath=outdir / "baseline_testing_scatter.png",
        target_label="Infant Mortality Rate",
        n_features=n_features_eff
    )

    with step("Permutation importance"):
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

    with step("Save artifacts"):
        pipe_path = outdir / "regressor_pipeline.joblib"
        joblib.dump(pipe, pipe_path)
        with open(outdir / "regressor_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    with step("Plots"):
        save_pred_vs_actual_plot(y_test.values, y_pred, outdir / "predicted_vs_actual.png")
        save_residuals_vs_fitted_plot(y_test.values, y_pred, outdir / "residuals_vs_fitted.png")
        save_residuals_hist_plot(y_test.values, y_pred, outdir / "residuals_hist.png")

    with step("Reports"):
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
