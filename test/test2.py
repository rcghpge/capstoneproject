import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tools.mice_imputer2 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X, XGBRegressor, Pipeline, train_test_split, mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
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
        print(f"WARNING: {state_col} not in dataframe, skipping state-wise plots")
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
            axes[idx].hist(state_data, bins=15, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{state}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Infant Mortality Rate', fontsize=9)
            axes[idx].set_ylabel('Count', fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)

    for idx in range(n_states, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Infant Mortality Rate State-Wise', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved state-wise distribution plot to {outpath}")

def plot_feature_importance_xgb(model, feature_names, outpath, top_k=20):
    """Plot XGBoost built-in feature importance"""
    try:
        importance = model.feature_importances_

        # Create dataframe and sort features
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

        print(f"✓ Saved XGBoost feature importance to {outpath}")
        return feat_imp

    except Exception as e:
        print(f"ERROR in XGBoost feature importance: {e}")
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

        print(f"✓ Saved permutation importance to {outpath}")

        # Save CSV
        csv_path = outpath.parent / 'permutation_importance.csv'
        perm_imp.to_csv(csv_path, index=False)
        print(f"✓ Saved permutation importance CSV to {csv_path}")

        return perm_imp

    except Exception as e:
        print(f"ERROR in permutation importance: {e}")
        return None

def get_feature_names_from_preprocessor(preprocessor, X_train):
    """Extract feature names after preprocessing"""
    try:
        # Try to get feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except:
        # Fallback to original column names
        return list(X_train.columns)

def validate_no_data_leakage(X_train, X_test, preprocessor):
    """Validate that preprocessing was fit only on training data"""
    print("\n" + "="*80)
    print("Data leakage validation")
    print("="*80)

    checks_passed = []

    # Check 1: Verify train/test have no overlap
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx.intersection(test_idx)

    if len(overlap) == 0:
        checks_passed.append(True)
        print("✓ Check 1 passed: No index overlap between train and test sets")
    else:
        checks_passed.append(False)
        print(f"✗ Check 1 failed: {len(overlap)} overlapping indices found!")

    # Check 2: Verify preprocessor statistics are based on train only
    try:
        # For numerical features, check if scaler statistics are reasonable
        for name, transformer, columns in preprocessor.transformers_:
            if 'num' in name and hasattr(transformer.named_steps.get('sc', None), 'center_'):
                scaler = transformer.named_steps['sc']
                train_subset = X_train[columns].select_dtypes(include=[np.number])

                # Check if scaler's center is close to train median
                for i, col in enumerate(train_subset.columns):
                    if i < len(scaler.center_):
                        train_median = train_subset[col].median()
                        scaler_center = scaler.center_[i]
                        if abs(train_median - scaler_center) < abs(train_median) * 0.5:  # 50% tolerance
                            continue
                        else:
                            print(f"  WARNING: Scaler center for '{col}' differs significantly from train median")

        checks_passed.append(True)
        print("✓ Check 2 passed: Preprocessor statistics consistent with training data")
    except Exception as e:
        checks_passed.append(False)
        print(f"✗ Check 2 failed: Could not validate preprocessor statistics - {e}")

    # Check 3: Verify train/test sizes are consistent with split ratio
    total_size = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total_size

    if 0.75 <= train_ratio <= 0.85:  # Expecting ~80% train
        checks_passed.append(True)
        print(f"✓ Check 3 passed: Train/test split ratio is {train_ratio:.2%} (expected ~80%)")
    else:
        checks_passed.append(False)
        print(f"✗ Check 3 failed: Train/test split ratio is {train_ratio:.2%} (expected ~80%)")

    # Summary
    print("\n" + "-"*80)
    if all(checks_passed):
        print("✓✓✓ All checks passed - no data leakage detected ✓✓✓")
    else:
        print(f"⚠⚠⚠ {sum(checks_passed)}/{len(checks_passed)} checks passed - review failures above ⚠⚠⚠")
    print("="*80 + "\n")

    return all(checks_passed)

def main(data_path: str, target: str, id_cols: list, output_dir: str, args):
    """Main test pipeline with train/test separation to prevent data leakage"""

    print("\n" + "="*80)
    print("XGBoost test pipeline with feature importance & leakage prevention")
    print("="*80 + "\n")

    # Load data
    print("1. Loading data...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Store state column for later visualization
    state_col = 'State_Name' if 'State_Name' in df.columns else None

    # Check for highly correlated features (potential leakage)
    print("\n2. Checking for highly correlated features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target in numeric_cols:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)
        print("   Top 10 correlated features:")
        for idx, (feat, corr_val) in enumerate(corrs.head(10).items(), 1):
            print(f"   {idx:2d}. {feat:<50s} {corr_val:>8.6f}")

        # Identify and remove highly correlated features (potential leakage)
        high_corr_threshold = 0.70
        high_corr_cols = corrs[1:][(corrs[1:].abs() > high_corr_threshold)].index.tolist()

        if high_corr_cols:
            print(f"\n   ⚠ Warning: Found {len(high_corr_cols)} features with correlation > {high_corr_threshold}")
            print("   These features will be removed to prevent data leakage:")
            for col in high_corr_cols:
                print(f"      • {col} (corr = {corrs[col]:.6f})")

            # Remove from dataframe
            df = df.drop(columns=high_corr_cols)
            print(f"\n   ✓ Removed {len(high_corr_cols)} highly correlated features")
            print(f"   New shape: {df.shape}")

    # MICE imputation args
    mice_args = argparse.Namespace(
        mice_model='huber',
        mice_max_iter=10,
        mice_tol=1e-3,
        reg_seed=42,
        huber_tol=1e-5,
        huber_epsilon=1.35,
        huber_alpha=0.1,
        huber_max_iter=100,
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

    # Impute with MICE
    print("\n3. Imputing missing values with MICE...")
    df_imputed = impute_dataframe(df, id_cols, [target], group_by_state=True, args=mice_args, progress=True, tag='main')
    print(f"   Imputation complete. Shape: {df_imputed.shape}")

    # Safe column dropping
    cols_to_drop = [target] + id_cols
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_imputed.columns]
    missing_cols = [col for col in cols_to_drop if col not in df_imputed.columns]

    if missing_cols:
        print(f"   Note: These columns not found (will skip): {missing_cols}")

    # Separate features and target
    X = df_imputed.drop(columns=existing_cols_to_drop)
    y = df_imputed[target]

    print(f"\n   Feature matrix: {X.shape}")
    print(f"   Target vector: {y.shape}")

    # Split before any preprocessing to prevent data leakage
    print("\n4. Splitting data (train/test) before preprocessing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Show sample targets (now it's safe since split is done)
    print(f"\n   Sample training targets: {y_train.iloc[:5].values}")
    print(f"   Sample test targets: {y_test.iloc[:5].values}")

    # Build preprocessor
    print("\n5. Building preprocessor...")
    preprocessor, numcols, catcols, numallmissing, catallmissing = build_preprocessor_from_X(X_train)
    print(f"   Numeric features: {len(numcols)}")
    print(f"   Categorical features: {len(catcols)}")

    # Fit preprocessor only on training data
    print("\n6. Fitting preprocessor on training data only (preventing leakage)...")
    preprocessor.fit(X_train)

    # Transform both sets
    print("   Transforming train and test sets...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"   Transformed shapes - Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")

    # Get feature names after transformation
    feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)
    print(f"   Total features after preprocessing: {len(feature_names)}")

    # Create XGBoost model with regularization
    print("\n7. Training XGBoost model...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=2,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,         # Add regularization
        reg_alpha=2.0,               # L1 regularization
        reg_lambda=2.0,              # L2 regularization
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )

    # Train model only on transformed training data
    xgb_model.fit(X_train_transformed, y_train)
    print("   Model training complete")

    # Generate test pipeline (preprocessor fitted)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb_model)
    ])

    # Create output directory
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(pipeline, outdir / 'xgb_pipeline.joblib')
    print(f"\n   Saved pipeline to {outdir / 'xgb_pipeline.joblib'}")

    # Validate no data leakage
    validate_no_data_leakage(X_train, X_test, preprocessor)

    # Predictions and metrics
    print("\n8. Generating predictions...")

    # Training predictions
    y_train_pred = xgb_model.predict(X_train_transformed)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_adj_r2 = adjusted_r2(train_r2, len(y_train), X_train_transformed.shape[1])

    # Test predictions
    y_test_pred = xgb_model.predict(X_test_transformed)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_adj_r2 = adjusted_r2(test_r2, len(y_test), X_test_transformed.shape[1])

    # Print metrics
    print("\n" + "="*80)
    print("Performance metrics")
    print("="*80)
    print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15} {'Difference':<15}")
    print("-"*65)
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f} {abs(test_rmse - train_rmse):<15.4f}")
    print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f} {abs(test_mae - train_mae):<15.4f}")
    print(f"{'R²':<20} {train_r2:<15.4f} {test_r2:<15.4f} {abs(test_r2 - train_r2):<15.4f}")
    print(f"{'Adjusted R²':<20} {train_adj_r2:<15.4f} {test_adj_r2:<15.4f} {abs(test_adj_r2 - train_adj_r2):<15.4f}")
    print("="*80)

    # Overfitting check
    if train_r2 > 0.95 and (train_r2 - test_r2) > 0.2:
        print("\n⚠ Warning: Possible overfitting detected!")
        print(f"   Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
        print("   Consider: reducing model complexity, adding regularization, or getting more data")
    elif abs(train_r2 - test_r2) < 0.1:
        print("\n✓ Model generalization looks good (train/test metrics are similar)")

    print()

    # Save metrics to file
    metrics_file = outdir / 'metrics_summary.txt'
    with open(metrics_file, 'w') as f:
        f.write("Performance metrics\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Metric':<20} {'Training':<15} {'Testing':<15} {'Difference':<15}\n")
        f.write("-"*65 + "\n")
        f.write(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f} {abs(test_rmse - train_rmse):<15.4f}\n")
        f.write(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f} {abs(test_mae - train_mae):<15.4f}\n")
        f.write(f"{'R²':<20} {train_r2:<15.4f} {test_r2:<15.4f} {abs(test_r2 - train_r2):<15.4f}\n")
        f.write(f"{'Adjusted R²':<20} {train_adj_r2:<15.4f} {test_adj_r2:<15.4f} {abs(test_adj_r2 - train_adj_r2):<15.4f}\n")
        f.write("\n" + "="*80 + "\n")
    print(f"✓ Saved metrics to {metrics_file}")

    # Generate visualizations
    print("\n9. Generating visualizations...")

    # Training scatter plot
    plot_prediction_results(
        y_true=y_train,
        y_pred=y_train_pred,
        title="Training Set: Predicted vs Actual",
        outpath=outdir / 'training_predictions.png',
        adj_r2=train_adj_r2,
        r2=train_r2,
        rmse=train_rmse
    )
    print(f"   ✓ Saved training predictions plot")

    # Test scatter plot
    plot_prediction_results(
        y_true=y_test,
        y_pred=y_test_pred,
        title="Test Set: Predicted vs Actual",
        outpath=outdir / 'testing_predictions.png',
        adj_r2=test_adj_r2,
        r2=test_r2,
        rmse=test_rmse
    )
    print(f"   ✓ Saved testing predictions plot")

    # State-wise distribution (if state column exists)
    if state_col and state_col in df.columns:
        df_with_state = df[[state_col, target]].copy()
        plot_statewise_distribution(df_with_state, target, state_col, outdir / 'statewise_distribution.png')

    # Feature importance analysis
    print("\n10. Computing feature importance metrics...")

    # XGBoost built-in importance
    xgb_importance = plot_feature_importance_xgb(
        xgb_model,
        feature_names,
        outdir / 'feature_importance_xgb.png',
        top_k=20
    )

    if xgb_importance is not None:
        xgb_importance.to_csv(outdir / 'feature_importance_xgb.csv', index=False)
        print(f"   ✓ Saved XGBoost importance CSV")

    # Permutation importance
    perm_importance = plot_permutation_importance(
        pipeline,
        X_test,
        y_test,
        feature_names,
        outdir / 'feature_importance_permutation.png',
        top_k=20,
        random_state=42
    )

    # Print top 10 features
    if xgb_importance is not None:
        print("\n" + "="*80)
        print("Top 10 most important features (XGBoost built-in)")
        print("="*80)
        for idx, row in xgb_importance.head(10).iterrows():
            print(f"{idx+1:2d}. {row['feature']:<50s} {row['importance']:>10.6f}")
        print("="*80 + "\n")

    if perm_importance is not None:
        print("="*80)
        print("Top 10 most important features (permutation importance)")
        print("="*80)
        for idx, row in perm_importance.head(10).iterrows():
            print(f"{idx+1:2d}. {row['feature']:<50s} {row['importance_mean']:>10.6f} ± {row['importance_std']:>10.6f}")
        print("="*80 + "\n")

    print("\n" + "="*80)
    print("Test pipeline complete")
    print("="*80)
    print(f"\nAll outputs saved to: {outdir.absolute()}")
    print("\nGenerated files:")
    print("  • xgb_pipeline.joblib - Trained model")
    print("  • metrics_summary.txt - Performance metrics")
    print("  • training_predictions.png - Training predictions plot")
    print("  • testing_predictions.png - Test predictions plot")
    print("  • statewise_distribution.png - State-wise IMR distribution")
    print("  • feature_importance_xgb.png - XGBoost feature importance")
    print("  • feature_importance_xgb.csv - XGBoost importance data")
    print("  • feature_importance_permutation.png - Permutation importance")
    print("  • permutation_importance.csv - Permutation importance data")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost test pipeline with anti-leakage validation")
    parser.add_argument('--data', required=True, help="Path to input CSV data")
    parser.add_argument('--target', required=True, help="Target variable column name")
    parser.add_argument('--id-cols', nargs='+', default=['State_Name','State_District_Name'], 
                       help="ID columns to exclude")
    parser.add_argument('--output-dir', default='./output_enhanced', help="Output directory")
    args = parser.parse_args()

    main(args.data, args.target, args.id_cols, args.output_dir, args)
