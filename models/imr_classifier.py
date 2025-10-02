#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
IMR Classification Model (scikit-learn) — with plots

Usage:
python imr_classifier.py \
  --data path/to/data.csv \
  --target "YY_Infant_Mortality_Rate_Imr_Total_Person" \
  --id-cols State_Name State_District_Name \
  --quantile 0.50 \ # or --threshold 12 (% vs numeric IMR metric for High (1) or Low (0))
  --outdir ./imr_classification_model

Outputs:
  - classifier_pipeline.joblib
  - classifier_metrics.json
  - roc_curve.png
  - pr_curve.png
  - confusion_matrix.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import joblib


def parse_args():
    p = argparse.ArgumentParser(description="Train a binary IMR classifier (separate script) with metrics & plots.")
    p.add_argument("--data", required=True, help="Path to .csv or .parquet dataset.")
    p.add_argument("--target", required=True, help="Numeric IMR target column to threshold for classification.")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID/index columns to exclude from features.")
    p.add_argument("--threshold", type=float, default=None, help="Fixed IMR threshold (y>=thr => 1).")
    p.add_argument("--quantile", type=float, default=None, help="Use y quantile as threshold (overrides --threshold).")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--outdir", default="./artifacts_cls", help="Output directory.")
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


def detect_types(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


def save_roc_plot(y_true, y_score, outpath: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_pr_plot(y_true, y_score, outpath: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_confusion_matrix_plot(cm: np.ndarray, outpath: Path, class_labels=("Low", "High")):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df(args.data)
    X, y = split_Xy(df, args.target, args.id_cols)
    num_cols, cat_cols = detect_types(X)
    pre = build_preprocessor(num_cols, cat_cols)

    # Decision threshold
    if args.quantile is not None:
        thr = float(np.nanquantile(y, args.quantile))
    elif args.threshold is not None:
        thr = float(args.threshold)
    else:
        thr = float(np.nanmedian(y))

    y_bin = (y >= thr).astype(int)

    unique, counts = np.unique(y_bin, return_counts=True)
    print("Class distribution (full):", dict(zip(unique.tolist(), counts.tolist())))
    if len(unique) < 2:
        print("Only one class after thresholding. Adjusting threshold...")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=args.test_size, random_state=args.random_state, stratify=y_bin
    )

    unique_tr, cnt_tr = np.unique(y_train, return_counts=True)
    unique_te, cnt_te = np.unique(y_test, return_counts=True)
    print("Train class dist:", dict(zip(unique_tr.tolist(), cnt_tr.tolist())))
    print("Test class dist:", dict(zip(unique_te.tolist(), cnt_te.tolist())))

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred))
    roc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "threshold": thr,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "num_features_numeric": len(num_cols),
        "num_features_categorical": len(cat_cols)
    }

    # Save artifacts
    pipe_path = outdir / "classifier_pipeline.joblib"
    joblib.dump(pipe, pipe_path)

    with open(outdir / "classifier_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    save_roc_plot(y_test, y_proba, outdir / "roc_curve.png")
    save_pr_plot(y_test, y_proba, outdir / "pr_curve.png")
    save_confusion_matrix_plot(cm, outdir / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {pipe_path.resolve()}")
    print(f"Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
