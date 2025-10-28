#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Pipeline with Optional Cross-Validation
Supports both K-Fold CV and simple train/test split (baseline mode)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tools.mice_imputer2 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X, XGBRegressor, Pipeline, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def adjusted_r2(r2, n, k):
    """Calculate adjusted R-squared"""
    if k >= n - 1:
        return np.nan
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))


def plot_prediction_results(y_true, y_pred, title, outpath, adj_r2, r2, rmse):
    """Generate prediction scatter plots with metrics"""
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(y_true)
    plt.plot(range(len(y_true)), y_pred[sorted_idx], label="Predicted Value", linewidth=2)
    plt.scatter(range(len(y_true)), np.array(y_true)[sorted_idx], color='k', s=15, label="True Value")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Sample", fontsize=12)
    plt.ylabel("Infant Mortality Rate", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    metrics = f"Metrics:\nR²={r2:.4f}\nAdj R²={adj_r2:.4f}\nRMSE={rmse:.4f}"
    plt.gca().text(1.02, 0.97, metrics, transform=plt.gca().transAxes, fontsize=11, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance_xgb(model, feature_names, outpath, topk=20):
    """Plot XGBoost built-in feature importance"""
    try:
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        feat_imp_top = feat_imp.head(topk)
        
        plt.figure(figsize=(10, max(6, 0.4 * len(feat_imp_top))))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp_top)))
        plt.barh(range(len(feat_imp_top)), feat_imp_top['importance'], color=colors)
        plt.yticks(range(len(feat_imp_top)), feat_imp_top['feature'], fontsize=9)
        plt.xlabel("Feature Importance", fontsize=12, fontweight='bold')
        plt.title(f"XGBoost Top {topk} Features (Built-in Importance)", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        return feat_imp
    except Exception as e:
        print(f"ERROR in XGBoost feature importance: {e}")
        return None


def get_feature_names_from_preprocessor(preprocessor, X_train):
    """Extract feature names after preprocessing"""
    try:
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except:
        return list(X_train.columns)


def run_baseline_split(df, target, id_cols, mice_args, xgb_params, test_size=0.2, random_state=42, high_corr_threshold=0.70):
    """Run a simple train/test split without cross-validation (baseline)"""
    print("=" * 80)
    print("BASELINE MODE: Single Train/Test Split (No CV)")
    print("=" * 80)
    
    # Remove highly correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target in numeric_cols:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)
        print("\nTop 10 features correlated with target:")
        for idx, (feat, corr_val) in enumerate(corrs.head(10).items(), 1):
            print(f"  {idx:2d}. {feat:60s} {corr_val:8.6f}")
        
        high_corr_cols = corrs[(corrs != 1) & (corrs.abs() >= high_corr_threshold)].index.tolist()
        if high_corr_cols:
            print(f"\nRemoving {len(high_corr_cols)} features with correlation >= {high_corr_threshold}")
            for col in high_corr_cols:
                print(f"  {col} (corr: {corrs[col]:.6f})")
            df = df.drop(columns=high_corr_cols)
            print(f"Shape after correlation filtering: {df.shape}")
        else:
            print(f"\nNo features with correlation >= {high_corr_threshold} to remove.")
    
    # Split data
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"\nTrain size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Impute
    print("\nImputing train data...")
    df_train_imp = impute_dataframe(
        df_train, id_cols, target, group_by_state=True, 
        args=mice_args, progress=False, tag="train"
    )
    print("Imputing test data...")
    df_test_imp = impute_dataframe(
        df_test, id_cols, target, group_by_state=True, 
        args=mice_args, progress=False, tag="test"
    )
    
    # Prepare features and target
    cols_to_drop = [target] + [col for col in id_cols if col in df_train_imp.columns]
    X_train = df_train_imp.drop(columns=cols_to_drop)
    y_train = df_train_imp[target]
    X_test = df_test_imp.drop(columns=cols_to_drop)
    y_test = df_test_imp[target]
    
    # Build preprocessor and transform
    print("\nBuilding preprocessor and transforming data...")
    preprocessor, _, _, _, _ = build_preprocessor_from_X(X_train)
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    
    # Train model
    print("Training XGBoost model...")
    model = XGBRegressor(**xgb_params)
    model.fit(X_train_t, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_t)
    y_test_pred = model.predict(X_test_t)
    
    # Compute metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_adj_r2 = adjusted_r2(train_r2, len(y_train), X_train_t.shape[1])
    test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test_t.shape[1])
    
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE:  {train_mae:.4f}, Test MAE:  {test_mae:.4f}")
    print(f"Train R²:   {train_r2:.4f}, Test R²:   {test_r2:.4f}")
    print(f"Train Adj R²: {train_adj_r2:.4f}, Test Adj R²: {test_adj_r2:.4f}")
    print("=" * 80)
    
    results = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_adj_r2': train_adj_r2,
        'test_adj_r2': test_adj_r2
    }
    
    feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)
    
    return results, model, preprocessor, X_train, X_test, y_train, y_test, feature_names


def cross_validate_with_cv(df, target, id_cols, mice_args, xgb_params, n_splits=5, 
                           random_state=42, high_corr_threshold=0.70, use_stratified=False):
    """Run cross-validation with data splitting and leakage prevention"""
    print("=" * 80)
    print(f"Cross-validation with {n_splits} folds")
    print("=" * 80)
    
    # Remove highly correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target in numeric_cols:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)
        print("\nTop 10 features correlated with target:")
        for idx, (feat, corr_val) in enumerate(corrs.head(10).items(), 1):
            print(f"  {idx:2d}. {feat:60s} {corr_val:8.6f}")
        
        high_corr_cols = corrs[(corrs != 1) & (corrs.abs() >= high_corr_threshold)].index.tolist()
        if high_corr_cols:
            print(f"\nRemoving {len(high_corr_cols)} features with correlation >= {high_corr_threshold}")
            for col in high_corr_cols:
                print(f"  {col} (corr: {corrs[col]:.6f})")
            df = df.drop(columns=high_corr_cols)
            print(f"Shape after correlation filtering: {df.shape}")
        else:
            print(f"\nNo features with correlation >= {high_corr_threshold} to remove.")
    
    # Initialize K-Fold
    if use_stratified:
        df_target_bin = pd.qcut(df[target], q=3, labels=False, duplicates='drop')
        stratify_labels = df_target_bin.values
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(df, stratify_labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(df)
    
    cv_metrics = {
        'train_rmse': [], 'test_rmse': [],
        'train_mae': [], 'test_mae': [],
        'train_r2': [], 'test_r2': [],
        'train_adj_r2': [], 'test_adj_r2': []
    }
    
    final_model = None
    final_preprocessor = None
    final_X_train = None
    final_X_test = None
    final_y_train = None
    final_y_test = None
    final_feature_names = None
    
    fold_num = 1
    for train_idx, test_idx in splits:
        print("=" * 80)
        print(f"Fold {fold_num}/{n_splits}")
        print("=" * 80)
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
        
        print("Imputing train fold...")
        df_train_imp = impute_dataframe(
            df_train, id_cols, target, group_by_state=True, 
            args=mice_args, progress=False, tag=f"fold{fold_num}-train"
        )
        print("Imputing test fold...")
        df_test_imp = impute_dataframe(
            df_test, id_cols, target, group_by_state=True, 
            args=mice_args, progress=False, tag=f"fold{fold_num}-test"
        )
        
        cols_to_drop = [target] + [col for col in id_cols if col in df_train_imp.columns]
        X_train = df_train_imp.drop(columns=cols_to_drop)
        y_train = df_train_imp[target]
        X_test = df_test_imp.drop(columns=cols_to_drop)
        y_test = df_test_imp[target]
        
        preprocessor, _, _, _, _ = build_preprocessor_from_X(X_train)
        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        
        model = XGBRegressor(**xgb_params)
        model.fit(X_train_t, y_train)
        
        y_train_pred = model.predict(X_train_t)
        y_test_pred = model.predict(X_test_t)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_adj_r2 = adjusted_r2(train_r2, len(y_train), X_train_t.shape[1])
        test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test_t.shape[1])
        
        print(f"\nFold {fold_num} Results:")
        print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"  Train MAE:  {train_mae:.4f}, Test MAE:  {test_mae:.4f}")
        print(f"  Train R²:   {train_r2:.4f}, Test R²:   {test_r2:.4f}")
        print(f"  Train Adj R²: {train_adj_r2:.4f}, Test Adj R²: {test_adj_r2:.4f}")
        
        cv_metrics['train_rmse'].append(train_rmse)
        cv_metrics['test_rmse'].append(test_rmse)
        cv_metrics['train_mae'].append(train_mae)
        cv_metrics['test_mae'].append(test_mae)
        cv_metrics['train_r2'].append(train_r2)
        cv_metrics['test_r2'].append(test_r2)
        cv_metrics['train_adj_r2'].append(train_adj_r2)
        cv_metrics['test_adj_r2'].append(test_adj_r2)
        
        if fold_num == n_splits:
            final_model = model
            final_preprocessor = preprocessor
            final_X_train = X_train
            final_X_test = X_test
            final_y_train = y_train
            final_y_test = y_test
            final_feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)
        
        fold_num += 1
    
    print("\n" + "=" * 80)
    print("Cross-validation summary (mean ± std)")
    print("=" * 80)
    for metric_name, values in cv_metrics.items():
        valid_values = [v for v in values if not (v is None or np.isnan(v))]
        if valid_values:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            print(f"{metric_name:15s} {mean_val:7.4f} ± {std_val:7.4f}")
        else:
            print(f"{metric_name:15s} No valid values")
    print("=" * 80)
    
    return cv_metrics, final_model, final_preprocessor, final_X_train, final_X_test, final_y_train, final_y_test, final_feature_names


def main(data_path, target, id_cols, output_dir, n_cv_folds=5, cv_random_state=42, 
         corr_threshold=0.70, use_stratified=False, baseline_mode=False, test_size=0.2):
    """Main pipeline with optional cross-validation or baseline mode"""
    print("=" * 80)
    if baseline_mode:
        print("XGBoost Pipeline - BASELINE MODE (No CV)")
    else:
        print("XGBoost Pipeline - CROSS-VALIDATION MODE")
    print("=" * 80)
    
    if not baseline_mode:
        print(f"CV Config: {n_cv_folds} folds, random_state={cv_random_state}")
    else:
        print(f"Baseline Config: test_size={test_size}, random_state={cv_random_state}")
    print(f"Correlation threshold: {corr_threshold}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # MICE imputation args
    mice_args = argparse.Namespace(
        mice_model='huber',
        mice_max_iter=10,
        mice_tol=1e-4,
        reg_seed=42,
        huber_tol=1e-5,
        huber_epsilon=1.55,
        huber_alpha=0.75,
        huber_max_iter=100,
        impute_targets='regression',
        allow_leakage=False,
        impute_categorical=True,
        add_flags=True,
        no_groupby_state=False,
        no_flags=False,
        no_cat_impute=False,
        target=target,
        progress=False
    )
    
    # XGBoost params
    xgb_params = {
        'n_estimators': 25,
        'learning_rate': 0.01,
        'max_depth': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'reg_alpha': 5.0,
        'reg_lambda': 5.0,
        'tree_method': 'gpu_hist',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    print(f"XGBoost tree_method: {xgb_params.get('tree_method', 'not set')}")
    
    # Run baseline or CV
    if baseline_mode:
        results, final_model, final_preprocessor, final_X_train, final_X_test, final_y_train, final_y_test, final_feature_names = run_baseline_split(
            df=df,
            target=target,
            id_cols=id_cols,
            mice_args=mice_args,
            xgb_params=xgb_params,
            test_size=test_size,
            random_state=cv_random_state,
            high_corr_threshold=corr_threshold
        )
    else:
        cv_results, final_model, final_preprocessor, final_X_train, final_X_test, final_y_train, final_y_test, final_feature_names = cross_validate_with_cv(
            df=df,
            target=target,
            id_cols=id_cols,
            mice_args=mice_args,
            xgb_params=xgb_params,
            n_splits=n_cv_folds,
            random_state=cv_random_state,
            high_corr_threshold=corr_threshold,
            use_stratified=use_stratified
        )
        results = cv_results
    
    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if baseline_mode:
        results_df = pd.DataFrame([results])
        results_df.to_csv(out_dir / 'baseline_results.csv', index=False)
        print(f"\nSaved baseline results to {out_dir / 'baseline_results.csv'}")
    else:
        cv_results_df = pd.DataFrame(results)
        cv_results_df.to_csv(out_dir / 'cv_results.csv', index=False)
        print(f"\nSaved CV results to {out_dir / 'cv_results.csv'}")
    
    # Generate visualizations
    print("\nGenerating visualizations from final model...")
    X_train_t = final_preprocessor.transform(final_X_train)
    X_test_t = final_preprocessor.transform(final_X_test)
    y_train_pred = final_model.predict(X_train_t)
    y_test_pred = final_model.predict(X_test_t)
    
    train_r2 = r2_score(final_y_train, y_train_pred)
    test_r2 = r2_score(final_y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(final_y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(final_y_test, y_test_pred))
    train_adj_r2 = adjusted_r2(train_r2, len(final_y_train), X_train_t.shape[1])
    test_adj_r2 = adjusted_r2(test_r2, len(final_y_test), X_test_t.shape[1])
    
    plot_prediction_results(
        y_true=final_y_train, y_pred=y_train_pred,
        title="Training Set: Predicted vs Actual",
        outpath=out_dir / 'training_predictions.png',
        adj_r2=train_adj_r2, r2=train_r2, rmse=train_rmse
    )
    print("  Saved training predictions plot")
    
    plot_prediction_results(
        y_true=final_y_test, y_pred=y_test_pred,
        title="Test Set: Predicted vs Actual",
        outpath=out_dir / 'testing_predictions.png',
        adj_r2=test_adj_r2, r2=test_r2, rmse=test_rmse
    )
    print("  Saved testing predictions plot")
    
    # Feature importance
    xgb_importance = plot_feature_importance_xgb(
        final_model, final_feature_names, 
        out_dir / 'feature_importance_xgb.png', topk=20
    )
    if xgb_importance is not None:
        xgb_importance.to_csv(out_dir / 'feature_importance_xgb.csv', index=False)
        print("  Saved XGBoost feature importance plot and CSV")
    
    print("=" * 80)
    if baseline_mode:
        print("XGBoost Baseline Pipeline Complete")
    else:
        print("XGBoost Cross-Validation Pipeline Complete")
    print("=" * 80)
    print(f"All outputs saved to {out_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Pipeline with Optional Cross-Validation")
    parser.add_argument('--data', required=True, help="Path to input CSV")
    parser.add_argument('--target', required=True, help="Target variable")
    parser.add_argument('--id-cols', nargs='+', default=['State_Name', 'State_District_Name'], help="ID columns to exclude")
    parser.add_argument('--output-dir', default='./output_cv', help="Output directory")
    parser.add_argument('--cv-folds', type=int, default=5, help="Number of CV folds (ignored in baseline mode)")
    parser.add_argument('--cv-random-state', type=int, default=42, help="Random state for CV/split")
    parser.add_argument('--corr-threshold', type=float, default=0.70, help="Correlation threshold")
    parser.add_argument('--stratified', action='store_true', help="Use stratified KFold (CV mode only)")
    parser.add_argument('--baseline', action='store_true', help="Run baseline mode (no CV, simple train/test split)")
    parser.add_argument('--test-size', type=float, default=0.2, help="Test size for baseline mode (default 0.2)")
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        target=args.target,
        id_cols=args.id_cols,
        output_dir=args.output_dir,
        n_cv_folds=args.cv_folds,
        cv_random_state=args.cv_random_state,
        corr_threshold=args.corr_threshold,
        use_stratified=args.stratified,
        baseline_mode=args.baseline,
        test_size=args.test_size
    )
