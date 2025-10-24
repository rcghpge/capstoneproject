import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tools.mice_imputer2 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X, XGBRegressor, Pipeline, train_test_split, mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

def plot_prediction_results(y_true, y_pred, title, outpath, adj_r2, r2, rmse):
    plt.figure(figsize=(10,6))
    sorted_idx = np.argsort(y_true)
    plt.plot(range(len(y_true)), y_pred[sorted_idx], label="Predicted Value")
    plt.scatter(range(len(y_true)), np.array(y_true)[sorted_idx], color='k', s=15, label="True Value")
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Infant Mortality Rate')
    plt.legend(loc='upper left')
    metrics = f"Metrics:\nR^2: {r2:.2f}\nAdjusted R^2: {adj_r2:.2f}\nRMSE: {rmse:.2f}"
    plt.gca().text(1.02, 0.97, metrics, transform=plt.gca().transAxes, fontsize=11, va='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.clf()

def main(data_path: str, target: str, id_cols: list, output_dir: str, args):
    df = pd.read_csv(data_path)
    mice_args = argparse.Namespace(
        mice_model='bayesridge',
        mice_max_iter=10,
        mice_tol=1e-3,
        reg_seed=42,
        impute_targets='regression',
        allowleakage=False,
        imputecategorical=True,
        add_flags=True,
        nogroupbystate=False,
        noflags=False,
        nocatimpute=False,
        target=[target],
        progress=True
    )
    df_imputed = impute_dataframe(df, id_cols, [target], group_by_state=True, args=mice_args, progress=True, tag='main')
    X = df_imputed.drop(columns=[target] + id_cols)
    y = df_imputed[target]
    preprocessor, numcols, catcols, numallmissing, catallmissing = build_preprocessor_from_X(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb_model)
    ])

    X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    pipeline.named_steps['preprocessor'].fit(X_train_inner)
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train_inner)
    X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val)

    pipeline.named_steps['regressor'].fit(
        X_train_transformed, y_train_inner
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, outdir / 'xgb_pipeline.joblib')

    X_train_full_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    y_train_pred = pipeline.named_steps['regressor'].predict(X_train_full_transformed)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_adj_r2 = adjusted_r2(train_r2, len(y_train), X_train.shape[1])

    plot_prediction_results(
        y_true=y_train, y_pred=y_train_pred, 
        title="Baseline Training Scatter Plot", 
        outpath=outdir / 'baseline_training_scatter.jpg',
        adj_r2=train_adj_r2, r2=train_r2, rmse=train_rmse
    )

    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    y_pred = pipeline.named_steps['regressor'].predict(X_test_transformed)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test.shape[1])

    plot_prediction_results(
        y_true=y_test, y_pred=y_pred, 
        title="Baseline Testing Scatter Plot", 
        outpath=outdir / 'baseline_testing_scatter.jpg',
        adj_r2=test_adj_r2, r2=test_r2, rmse=test_rmse
    )

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Train Adjusted R^2: {train_adj_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Test Adjusted R^2: {test_adj_r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined MICE + XGBoost Pipeline")
    parser.add_argument('--data', required=True, help="Path to input CSV data")
    parser.add_argument('--target', required=True, help="Target variable column name")
    parser.add_argument('--id-cols', nargs='+', default=['State_Name','State_District_Name'], help="ID columns to exclude")
    parser.add_argument('--output-dir', default='./output', help="Output directory for model and reports")
    args = parser.parse_args()
    main(args.data, args.target, args.id_cols, args.output_dir, args)
