import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost.callback import EarlyStopping
from tools.mice_imputer2 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X, XGBRegressor, Pipeline, train_test_split, mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

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
    callbacks = [EarlyStopping(rounds=100, save_best=True)]
    pipeline.named_steps['regressor'].fit(
        X_train_transformed, y_train_inner,
        #eval_set=[(X_val_transformed, y_val)],
        #callbacks=callbacks,
        verbose=False
    )
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    y_pred = pipeline.named_steps['regressor'].predict(X_test_transformed)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2: {r2:.4f}")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, outdir / 'xgb_pipeline.joblib')
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.savefig(outdir / 'predicted_vs_actual.png')
    plt.clf()
    residuals = y_test - y_pred
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title('Residuals Histogram')
    plt.savefig(outdir / 'residuals_hist.png')
    plt.clf()
    plt.scatter(pipeline.named_steps['regressor'].predict(X_test_transformed), residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Fitted')
    plt.savefig(outdir / 'residuals_vs_fitted.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined MICE + XGBoost Pipeline")
    parser.add_argument('--data', required=True, help="Path to input CSV data")
    parser.add_argument('--target', required=True, help="Target variable column name")
    parser.add_argument('--id-cols', nargs='+', default=['State_Name','State_District_Name'], help="ID columns to exclude")
    parser.add_argument('--output-dir', default='./output', help="Output directory for model and reports")
    args = parser.parse_args()
    main(args.data, args.target, args.id_cols, args.output_dir, args)
