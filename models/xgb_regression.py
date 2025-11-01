#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from xgboost import XGBRegressor

def parse_args():
    p = argparse.ArgumentParser(description="IMR XGBoost Regression with RFECV feature selection")
    p.add_argument("--data", required=True, help="Path to dataset CSV file")
    p.add_argument("--target", required=True, help="Target column to predict")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID columns to exclude from features")
    p.add_argument("--outdir", default="./artifacts_xgb_rfecv", help="Output directory")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random state")
    p.add_argument("--n-estimators", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    return p.parse_args()

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_ok = [c for c in num_cols if not X[c].isna().all()]
    cat_ok = [c for c in cat_cols if not X[c].isna().all()]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_ok),
        ("cat", cat_pipe, cat_ok)
    ])
    return preprocessor

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV data only
    if not args.data.endswith(".csv"):
        raise ValueError("Only CSV format supported in this script.")
    df = pd.read_csv(args.data)

    # Features/target split
    X = df.drop(columns=[args.target] + args.id_cols)
    y = df[args.target].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Preprocessing
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Setup XGBoost regressor
    xgb = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        verbosity=0
    )

    # RFECV for feature selection using XGB as estimator
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    selector = RFECV(
        estimator=xgb,
        step=10,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    selector.fit(X_train_processed, y_train)

    # Get selected features names
    rf_support = selector.get_support()
    feature_names = preprocessor.get_feature_names_out()
    selected_features = np.array(feature_names)[rf_support]

    # Transform train/test with selected features
    X_train_selected = selector.transform(X_train_processed)
    X_test_selected = selector.transform(X_test_processed)

    # Train final model on selected features
    final_xgb = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        verbosity=0
    )
    final_xgb.fit(X_train_selected, y_train)

    # Predictions and evaluation
    y_pred = final_xgb.predict(X_test_selected)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": evs,
        "selected_feature_count": len(selected_features),
    }

    # Save outputs
    np.save(outdir / "X_train_selected.npy", X_train_selected)
    np.save(outdir / "X_test_selected.npy", X_test_selected)
    np.save(outdir / "y_train.npy", y_train.to_numpy())
    np.save(outdir / "y_test.npy", y_test.to_numpy())
    np.save(outdir / "y_pred.npy", y_pred)
    with open(outdir / "regressor_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Selected features by RFECV:")
    for feat in selected_features:
        print(feat)
    print("\nRFECV estimator feature importances:")
    for feat, importance in zip(selected_features, selector.estimator_.feature_importances_):
        print(f"{feat}: {importance:.4f}")

    print("\nFinal XGBoost model feature importances:")
    for feat, importance in zip(selected_features, final_xgb.feature_importances_):
        print(f"{feat}: {importance:.4f}")

if __name__ == "__main__":
    main()
