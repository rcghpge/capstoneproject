import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tools.mice_imputer2 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X, XGBRegressor, Pipeline, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
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
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Infant Mortality Rate', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)

    metrics = f"Metrics:\nR²: {r2:.4f}\nAdj R²: {adj_r2:.4f}\nRMSE: {rmse:.4f}"
    plt.gca().text(1.02, 0.97, metrics, transform=plt.gca().transAxes, fontsize=11,
                   va='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_statewise_distribution(df, target, state_col, outpath):
    """Create state-wise histogram plots like Landon's"""
    if state_col not in df.columns:
        print(f"  WARNING: {state_col} not in dataframe, skipping state-wise plots")
        return

    states = df[state_col].unique()
    n_states = len(states)

    # Calculate grid dimensions
    n_cols = 3
    n_rows = int(np.ceil(n_states / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_states > 1 else [axes]

    for idx, state in enumerate(sorted(states)):
        state_data = df[df[state_col] == state][target].dropna()

        if len(state_data) > 0:
            axes[idx].hist(state_data, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{state}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Infant Mortality Rate', fontsize=9)
            axes[idx].set_ylabel('Count', fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)

    for idx in range(n_states, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Infant Mortality Rate Distribution by State', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance_xgb(model, feature_names, outpath, top_k=20):
    """Plot XGBoost built-in feature importance"""
    try:
        importance = model.feature_importances_

        # Create dataframe and sort
        feat_imp = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Get top K
        feat_imp_top = feat_imp.head(top_k)

        # Plot
        plt.figure(figsize=(10, max(6, 0.4 * len(feat_imp_top))))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp_top)))

        plt.barh(range(len(feat_imp_top)), feat_imp_top['importance'], color=colors)
        plt.yticks(range(len(feat_imp_top)), feat_imp_top['feature'], fontsize=9)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title(f'XGBoost Top {top_k} Features (Built-in Importance)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

        return feat_imp

    except Exception as e:
        print(f"  ERROR in XGBoost feature importance: {e}")
        return None

def plot_permutation_importance(pipeline, X_test, y_test, feature_names, outpath, top_k=20, random_state=42):
    """Calculate and plot permutation importance"""
    try:
        print("  Computing permutation importance (this may take a minute)...")

        sample_size = min(500, len(X_test))
        sample_idx = np.random.RandomState(random_state).choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_idx]
        y_sample = y_test.iloc[sample_idx]

        # Compute permutation importance
        perm_importance = permutation_importance(
            pipeline, X_sample, y_sample,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1
        )

        # Create dataframe
        perm_imp = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        # Get top K
        perm_imp_top = perm_imp.head(top_k)

        # Plot
        plt.figure(figsize=(10, max(6, 0.4 * len(perm_imp_top))))
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(perm_imp_top)))

        plt.barh(range(len(perm_imp_top)), perm_imp_top['importance_mean'],
                xerr=perm_imp_top['importance_std'], color=colors, alpha=0.8)
        plt.yticks(range(len(perm_imp_top)), perm_imp_top['feature'], fontsize=9)
        plt.xlabel('Permutation Importance (Mean ± Std)', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_k} Features - Permutation Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

        return perm_imp

    except Exception as e:
        print(f"  ERROR in permutation importance: {e}")
        return None

def get_feature_names_from_preprocessor(preprocessor, X_train):
    """Extract feature names after preprocessing"""
    try:
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except:
        return list(X_train.columns)

def cross_validate_with_cv(df, target, id_cols, mice_args, xgb_params,
                           n_splits=5, random_state=42, high_corr_threshold=0.70, use_stratified=False):
    """
    Run cross-validation with data splitting and leakage prevention
    """
    print(f"\n{'='*80}")
    print(f"Cross-validation with {n_splits} folds")
    print(f"{'='*80}\n")

    # Remove highly correlated features before splitting
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target in numeric_cols:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)

        # Print top 10 correlated features
        print("Top 10 features correlated with target:")
        for idx, (feat, corr_val) in enumerate(corrs.head(10).items(), 1):
            print(f"  {idx:2d}. {feat:<60s} {corr_val:>8.6f}")

        # Identify highly correlated features to remove
        high_corr_cols = corrs[1:][(corrs[1:].abs() >= high_corr_threshold)].index.tolist()

        if high_corr_cols:
            print(f"\n⚠ Removing {len(high_corr_cols)} features with |correlation| >= {high_corr_threshold}:")
            for col in high_corr_cols:
                print(f"  • {col} (corr = {corrs[col]:.6f})")

            df = df.drop(columns=high_corr_cols)
            print(f"\nNew shape after correlation filtering: {df.shape}\n")
        else:
            print(f"\nNo features with |correlation| >= {high_corr_threshold} to remove.\n")

    # Initialize KFold logic (standard KFold vs Stratified KFold)
    if use_stratified:
        # Create bins (e.g., 3 quantiles)
        df['target_bin'] = pd.qcut(df[target], q=3, labels=False, duplicates='drop')
        stratify_labels = df['target_bin'].values
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(df, stratify_labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kf.split(df)

    # Storage for metrics and final model artifacts
    cv_metrics = {
        'train_rmse': [], 'test_rmse': [],
        'train_mae': [], 'test_mae': [],
        'train_r2': [], 'test_r2': [],
        'train_adj_r2': [], 'test_adj_r2': []
    }

    # Store final fold for plotting
    final_model = None
    final_preprocessor = None
    final_X_train = None
    final_X_test = None
    final_y_train = None
    final_y_test = None
    final_feature_names = None

    fold_num = 1
    for train_idx, test_idx in splits:
        print(f"{'='*80}")
        print(f"Fold {fold_num}/{n_splits}")
        print(f"{'='*80}")

        # Split data
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

        # MICE imputation - implemented separately
        print("Imputing train fold...")
        df_train_imp = impute_dataframe(
            df_train, id_cols, [target],
            group_by_state=True, args=mice_args,
            progress=False, tag=f'fold{fold_num}_train'
        )

        print("Imputing test fold...")
        df_test_imp = impute_dataframe(
            df_test, id_cols, [target],
            group_by_state=True, args=mice_args,
            progress=False, tag=f'fold{fold_num}_test'
        )

        # Prepare features and target
        cols_to_drop = [target] + [col for col in id_cols if col in df_train_imp.columns]

        X_train = df_train_imp.drop(columns=cols_to_drop)
        y_train = df_train_imp[target]
        X_test = df_test_imp.drop(columns=cols_to_drop)
        y_test = df_test_imp[target]

        # Build and fit preprocessor on train only
        preprocessor, _, _, _, _ = build_preprocessor_from_X(X_train)
        preprocessor.fit(X_train)

        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # Train model
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

        # Display fold metrics
        print(f"\nFold {fold_num} Results:")
        print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"  Train MAE:  {train_mae:.4f}, Test MAE:  {test_mae:.4f}")
        print(f"  Train R²:   {train_r2:.4f}, Test R²:   {test_r2:.4f}")
        print(f"  Train Adj R²: {train_adj_r2}, Test Adj R²: {test_adj_r2}\n")

        # Store metrics
        cv_metrics['train_rmse'].append(train_rmse)
        cv_metrics['test_rmse'].append(test_rmse)
        cv_metrics['train_mae'].append(train_mae)
        cv_metrics['test_mae'].append(test_mae)
        cv_metrics['train_r2'].append(train_r2)
        cv_metrics['test_r2'].append(test_r2)
        cv_metrics['train_adj_r2'].append(train_adj_r2)
        cv_metrics['test_adj_r2'].append(test_adj_r2)

        # Store final fold for plotting
        if fold_num == n_splits:
            final_model = model
            final_preprocessor = preprocessor
            final_X_train = X_train
            final_X_test = X_test
            final_y_train = y_train
            final_y_test = y_test
            final_feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)

        fold_num += 1

    # Aggregate CV results
    print(f"\n{'='*80}")
    print("Cross-validation summary (mean ± std)")
    print(f"{'='*80}\n")

    for metric_name, values in cv_metrics.items():
        valid_values = [v for v in values if not (v is None or np.isnan(v))]
        if valid_values:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            print(f"{metric_name:15s}: {mean_val:7.4f} ± {std_val:7.4f}")
        else:
            print(f"{metric_name:15s}: No valid values")

    print(f"\n{'='*80}\n")

    return cv_metrics, final_model, final_preprocessor, final_X_train, final_X_test, final_y_train, final_y_test, final_feature_names

def main(data_path, target, id_cols, output_dir,
         n_cv_folds=5, cv_random_state=42, corr_threshold=0.70, use_stratified=False):
    """Main pipeline with cross-validation and visualization"""

    print(f"\n{'='*80}")
    print("XGBoost test pipeline with K-Fold Cross-Validation + Visualizations")
    print(f"{'='*80}")
    print(f"CV Config: {n_cv_folds} folds, random_state={cv_random_state}")
    print(f"Correlation threshold: {corr_threshold}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df_original = df.copy()  # Keep for state-wise plots
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

    # MICE imputation args
    mice_args = argparse.Namespace(
        mice_model='huber',
        mice_max_iter=10,
        mice_tol=1e-4,
        reg_seed=42,
        huber_tol=1e-5,
        huber_epsilon=1.55,
        huber_alpha=0.75,
        huber_max_iter=1000,
        impute_targets='regression', # median, regression
        allowleakage=False,
        imputecategorical=True,
        add_flags=True,
        nogroupbystate=False,
        noflags=False,
        nocatimpute=False,
        target=[target],
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

    # GPU checks
    print("XGBoost tree_method:", xgb_params.get("tree_method", "not set"))

    # Run cross-validation
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

    # Save CV results
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(outdir / 'cv_results.csv', index=False)
    print(f"✓ Saved CV results to {outdir / 'cv_results.csv'}")

    # Save summary statistics
    summary_file = outdir / 'cv_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Cross-Validation Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  CV Folds: {n_cv_folds}\n")
        f.write(f"  Random State: {cv_random_state}\n")
        f.write(f"  Correlation Threshold: {corr_threshold}\n\n")
        f.write("Metrics (mean ± std):\n")
        f.write("-"*80 + "\n")

        for metric_name, values in cv_results.items():
            valid_values = [v for v in values if not (v is None or np.isnan(v))]
            if valid_values:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                f.write(f"{metric_name:15s}: {mean_val:7.4f} ± {std_val:7.4f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✓ Saved CV summary to {summary_file}")

    # Generate visualizations using final fold
    print(f"\nGenerating visualizations from final fold...")

    # Compute final predictions for plotting
    X_train_t = final_preprocessor.transform(final_X_train)
    X_test_t = final_preprocessor.transform(final_X_test)
    y_train_pred = final_model.predict(X_train_t)
    y_test_pred = final_model.predict(X_test_t)

    # Compute final metrics
    train_r2 = r2_score(final_y_train, y_train_pred)
    test_r2 = r2_score(final_y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(final_y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(final_y_test, y_test_pred))
    train_adj_r2 = adjusted_r2(train_r2, len(final_y_train), X_train_t.shape[1])
    test_adj_r2 = adjusted_r2(test_r2, len(final_y_test), X_test_t.shape[1])

    # Training scatter plot
    plot_prediction_results(
        y_true=final_y_train,
        y_pred=y_train_pred,
        title="Training Set: Predicted vs Actual (Final Fold)",
        outpath=outdir / 'training_predictions.png',
        adj_r2=train_adj_r2,
        r2=train_r2,
        rmse=train_rmse
    )
    print(f"  ✓ Saved training predictions plot")

    # Test scatter plot
    plot_prediction_results(
        y_true=final_y_test,
        y_pred=y_test_pred,
        title="Test Set: Predicted vs Actual (Final Fold)",
        outpath=outdir / 'testing_predictions.png',
        adj_r2=test_adj_r2,
        r2=test_r2,
        rmse=test_rmse
    )
    print(f"  ✓ Saved testing predictions plot")

    # State-wise distribution (if state column exists)
    state_col = 'State_Name' if 'State_Name' in df_original.columns else None
    if state_col and target in df_original.columns:
        plot_statewise_distribution(df_original, target, state_col, outdir / 'statewise_distribution.png')
        print(f"  ✓ Saved state-wise distribution plot")

    # XGBoost built-in importance
    xgb_importance = plot_feature_importance_xgb(
        final_model,
        final_feature_names,
        outdir / 'feature_importance_xgb.png',
        top_k=20
    )
    if xgb_importance is not None:
        xgb_importance.to_csv(outdir / 'feature_importance_xgb.csv', index=False)
        print(f"  ✓ Saved XGBoost feature importance plot and CSV")

    # Create pipeline for permutation importance
    pipeline = Pipeline([
        ('preprocessor', final_preprocessor),
        ('regressor', final_model)
    ])

    # Permutation importance
    perm_importance = plot_permutation_importance(
        pipeline,
        final_X_test,
        final_y_test,
        final_feature_names,
        outdir / 'feature_importance_permutation.png',
        top_k=20,
        random_state=42
    )
    if perm_importance is not None:
        perm_importance.to_csv(outdir / 'permutation_importance.csv', index=False)
        print(f"  ✓ Saved permutation importance plot and CSV")

    print(f"\n{'='*80}")
    print("XGBoost Cross-validation test pipeline complete with visualizations")
    print(f"{'='*80}\n")
    print(f"All outputs saved to: {outdir.absolute()}\n")

if __name__ == "__main__":
    # CLI flags (runs in the terminal)
    parser = argparse.ArgumentParser(
        description="XGBoost test pipeline with K-Fold Cross-Validation"
    )
    parser.add_argument('--data', required=True, help="Path to input CSV")
    parser.add_argument('--target', required=True, help="Target variable")
    parser.add_argument('--id-cols', nargs='+',
                       default=['State_Name', 'State_District_Name'],
                       help="ID columns to exclude")
    parser.add_argument('--output-dir', default='./output_cv',
                       help="Output directory")

    parser.add_argument('--cv-folds', type=int, default=5,
                       help="Number of CV folds (default=5)")
    parser.add_argument('--cv-random-state', type=int, default=42,
                       help="Random state for CV (default=42)")
    parser.add_argument('--corr-threshold', type=float, default=0.70,
                       help="Correlation threshold (default=0.70)")
    parser.add_argument('--stratified', action='store_true',
                    help='Use stratified KFold with binned target for regression')

    args = parser.parse_args()

    main(
        data_path=args.data,
        target=args.target,
        id_cols=args.id_cols,
        output_dir=args.output_dir,
        n_cv_folds=args.cv_folds,
        cv_random_state=args.cv_random_state,
        corr_threshold=args.corr_threshold,
        use_stratified=args.stratified
    )
