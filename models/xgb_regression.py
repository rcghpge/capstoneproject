#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
"""
Example Usage:
python -m models.xgb_regression --data data/Key_indicator_districtwise.csv \
--target YY_Infant_Mortality_Rate_Imr_Total_Person --id-cols State_Name State_District_Name \
--outdir ./xgb
"""


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns

def plot_and_save(y_true, y_pred, outdir):
    Path(outdir).mkdir(exist_ok=True, parents=True)

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.savefig(Path(outdir) / 'predicted_vs_actual.png')
    plt.close()

    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--', color='r')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.savefig(Path(outdir) / 'residuals_vs_fitted.png')
    plt.close()

    plt.figure()
    plt.hist(residuals, bins=30, alpha=0.8)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    plt.savefig(Path(outdir) / 'residual_histogram.png')
    plt.close()

def plot_true_vs_pred(y_true, y_pred, outdir, subset_label, metrics):
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y_true)), y_true, color='black', label='True Value', s=20)
    plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Value')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'Baseline {subset_label} Scatter Plot')
    plt.legend()

    textstr = '\n'.join((
        f"R^2: {metrics['r2']:.2f}",
        f"Adjusted R^2: {metrics.get('adj_r2', 0):.2f}",
        f"RMSE: {metrics['rmse']:.2f}",
        f"MAE: {metrics['mae']:.2f}"
    ))
    plt.gca().text(0.95, 0.1, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    plt.tight_layout()
    plt.savefig(Path(outdir) / f'baseline_{subset_label.lower()}_scatter.png')
    plt.close()

def plot_feature_importances(importances, feature_names, outdir, top_n=10):
    top_n = min(top_n, len(importances), len(feature_names))
    idx = np.argsort(importances)[-top_n:]
    top_importances = importances[idx]
    top_features = np.array(feature_names)[idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_n), top_importances, align='center', color='steelblue')
    plt.yticks(range(top_n), top_features)
    plt.xlabel("Feature Importances")
    plt.title(f"Top {top_n} Model Features")
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'feature_importances.png')
    plt.close()

def plot_statewise_histogram(df, value_col, state_col, outdir):
    sns.histplot(data=df, x=value_col, hue=state_col, element="step", stat="count", common_norm=False, palette='bright', alpha=0.4)
    plt.title('Value Distribution State-Wise')
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'statewise_histogram.png')
    plt.close()

def plot_statewise_facets(df, value_col, state_col, outdir):
    g = sns.FacetGrid(df, col=state_col, col_wrap=3, height=3, aspect=1.5, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x=value_col, stat="count", bins=10, color='steelblue')
    g.set_titles("{col_name}")
    g.set_axis_labels("Value", "Count")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Value Distribution State-Wise')
    plt.savefig(Path(outdir) / 'statewise_facets.png')
    plt.close()

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

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

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
    xgb_model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        verbosity=0
    )
    xgb_model.fit(X_train_selected, y_train)
    y_pred_train = xgb_model.predict(X_train_selected)
    y_test_pred = xgb_model.predict(X_test_selected)

    # Predictions and evaluation
    y_pred = xgb_model.predict(X_test_selected)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    n_samples = len(y_test)
    n_features = len(selected_features)
    adj_r2 = adjusted_r2(r2, n_samples, n_features)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": evs,
        "selected_feature_count": n_features,
        "adj_r2": adj_r2,
    }

    # Save outputs
    np.save(outdir / "X_train_selected.npy", X_train_selected)
    np.save(outdir / "X_test_selected.npy", X_test_selected)
    np.save(outdir / "y_train.npy", y_train.to_numpy())
    np.save(outdir / "y_test.npy", y_test.to_numpy())
    np.save(outdir / "y_pred.npy", y_pred)

    with open(outdir / "regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(outdir / "regression_metrics.txt", "w") as f:
        f.write("Regression Metrics Summary\n")
        f.write("==========================\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    with open(outdir / "regression_report.txt", "w") as f:
        f.write("Model Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")

        f.write("Selected features by RFECV:\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
        f.write("\n")

        f.write("RFECV estimator feature importance:\n")
        for feat, importance in zip(selected_features, selector.estimator_.feature_importances_):
            f.write(f"{feat}: {importance:.4f}\n")
        f.write("\n")

        f.write("Final XGBoost model feature importance:\n")
        for feat, importance in zip(selected_features, xgb_model.feature_importances_):
            f.write(f"{feat}: {importance:.4f}\n")


    print("\nModel Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    print("\nSelected features by RFECV:")
    for feat in selected_features:
        print(feat)
    print("\nRFECV estimator feature importance:")
    for feat, importance in zip(selected_features, selector.estimator_.feature_importances_):
        print(f"{feat}: {importance:.4f}")

    print("\nFinal XGBoost model feature importance:")
    for feat, importance in zip(selected_features, xgb_model.feature_importances_):
        print(f"{feat}: {importance:.4f}")

    train_metrics = {
        "r2": r2_score(y_train, y_pred_train),
        "adj_r2": adjusted_r2(r2_score(y_train, y_pred_train), len(y_train), n_features),
        "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "mae": mean_absolute_error(y_train, y_pred_train),
    }

    test_metrics = {
        "r2": r2_score(y_test, y_pred),
        "adj_r2": adjusted_r2(r2_score(y_test, y_pred), len(y_test), n_features),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    plot_and_save(y_test, y_pred, outdir)
    plot_true_vs_pred(y_train, y_pred_train, outdir, "Train", train_metrics)
    plot_true_vs_pred(y_test, y_pred, outdir, "Test", test_metrics)

    feature_importances = xgb_model.feature_importances_  # or selector.estimator_.feature_importances_
    feature_names = selected_features
    plot_feature_importances(feature_importances, feature_names, outdir)
    plot_statewise_histogram(df, value_col=args.target, state_col="State_Name", outdir=outdir)
    plot_statewise_facets(df, value_col=args.target, state_col="State_Name", outdir=outdir)

if __name__ == "__main__":
    main()
