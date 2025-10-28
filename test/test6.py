import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.model_selection import LeaveOneOut, KFold
from xgboost import XGBRegressor
#from models.imr_xgb_regressor2 import XGBRegressor
from tools.mice_imputer3 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def adjusted_r2(r2, n, k):
    if k >= n - 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def run_loocv(df, target, id_cols, mice_args, xgb_params, corr_threshold=0.7, var_threshold=0.01, suppress_warnings=True):
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    if suppress_warnings:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    else:
        warnings.filterwarnings("default", category=UndefinedMetricWarning)

    print("Droping Prefix from Feature Column Names")
    df.columns = [re.sub(r'^[A-Z]{2}_', '', col) for col in df.columns]
    cols_after_drop = len(pd.Index(df.columns).unique())
    print(f"Feature Columns after prefix drop: {cols_after_drop}")

    # Remove constants and near-constant features
    feat_cols = df.select_dtypes(include=np.number).columns
    if len(df) == 0:
        raise ValueError("No numeric feature columns remain after dropping target and id columns and prefix drops.")

    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(df[feat_cols])
    kept_features = feat_cols[vt.get_support()]
    print(f"VarianceThreshold filter kept {len(kept_features)} features from {len(feat_cols)}")

    df_filtered = df[id_cols + list(kept_features) + [target]]

    loo = LeaveOneOut()
    train_r2s, test_r2s = [], []
    train_adj_r2s, test_adj_r2s = [], []
    fold_num = 0

    for train_idx, test_idx in loo.split(df_filtered):
        fold_num += 1
        df_train = df_filtered.iloc[train_idx].reset_index(drop=True)
        df_test = df_filtered.iloc[test_idx].reset_index(drop=True)

        # Impute
        df_train_imp = impute_dataframe(df_train, id_cols, [target], args=mice_args, progress=False, tag=f"loocv_{fold_num}_train")
        df_test_imp = impute_dataframe(df_test, id_cols, [target], args=mice_args, progress=False, tag=f"loocv_{fold_num}_test")

        drop_cols = [target] + id_cols
        X_train = df_train_imp.drop(columns=drop_cols)
        y_train = df_train_imp[target]
        X_test = df_test_imp.drop(columns=drop_cols)
        y_test = df_test_imp[target]

        # Preprocess
        preprocessor, *_ = build_preprocessor_from_X(X_train)
        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()

        # RFECV for robust feature selection in fold training data
        cv_inner = KFold(n_splits=min(5, len(y_train)))
        xgb_estimator = XGBRegressor(**xgb_params)
        rfecv = RFECV(estimator=xgb_estimator, step=1, cv=cv_inner, scoring='r2', n_jobs=-1)
        rfecv.fit(X_train_t, y_train)

        selected_mask = rfecv.support_
        selected_features = np.array(feature_names)[selected_mask]
        print(f"Fold {fold_num}: RFECV selected {len(selected_features)} features")

        # Reduced data
        X_train_r = X_train_t[:, selected_mask]
        X_test_r = X_test_t[:, selected_mask]

        # Train XGBoost on selected features
        model = XGBRegressor(**xgb_params)
        model.fit(X_train_r, y_train)

        train_r2 = model.score(X_train_r, y_train)
        test_r2 = model.score(X_test_r, y_test)
        y_test_pred = model.predict(X_test_r)


        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)

        # Store per fold
        fold_rmse_list.append(rmse)
        fold_mae_list.append(mae)

        # Print per fold diagnostics
        print(f"Fold {fold_num}: RMSE {rmse:.4f}, MAE {mae:.4f}")

        train_r2s.append(train_r2)
        test_r2s.append(test_r2)
        train_adj_r2s.append(adjusted_r2(train_r2, len(y_train), len(selected_features)))
        test_adj_r2s.append(adjusted_r2(test_r2, len(y_test), len(selected_features)))

        print(f"Fold {fold_num}: Train R2 {train_r2:.4f}, Test R2 {test_r2:.4f}")

    print(f"LOOCV Results:\nTrain R2: {np.mean(train_r2s):.4f} ± {np.std(train_r2s):.4f}\n"
          f"Test R2: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")

    return {
        'train_r2': train_r2s,
        'test_r2': test_r2s,
        'train_adj_r2': train_adj_r2s,
        'test_adj_r2': test_adj_r2s
    }

def main():
    parser = argparse.ArgumentParser(description="Robust LOOCV pipeline with RFECV feature selection")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--id-cols", nargs="+", required=True)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--var-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suppress-warnings", type=bool, default=True,
                    help="Suppress UndefinedMetricWarning for R2 when test set has less than two samples (default True)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    mice_args = argparse.Namespace(
        mice_model="huber", mice_max_iter=1000, mice_tol=1e-4, reg_seed=args.seed,
        huber_tol=1e-5, huber_epsilon=1.0, huber_alpha=0.5, huber_max_iter=1000,
        impute_targets="regression", allowleakage=False, imputecategorical=True,
        add_flags=True, nogroupbystate=False, noflags=False, nocatimpute=False,
        target=[args.target], progress=False
    )

    xgb_params = {
        "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 4,
        "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 10,
        "reg_alpha": 2.0, "reg_lambda": 2.0, "tree_method": "hist",
        "n_jobs": -1, "random_state": args.seed, "verbosity": 0
    }

    results = run_loocv(
        df, args.target, args.id_cols, mice_args, xgb_params,
        corr_threshold=args.corr_threshold, var_threshold=args.var_threshold
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save results
    import json
    with open(output_dir / "loocv_results.json", "w") as f:
        json.dump(results, f)

    print(f"Saved LOOCV results JSON to {output_dir}")

if __name__ == "__main__":
    main()
