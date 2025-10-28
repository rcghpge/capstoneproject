import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tools.mice_imputer3 import impute_dataframe
from models.imr_xgb_regressor2 import build_preprocessor_from_X, XGBRegressor, Pipeline, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
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
    """Create state-wise histogram plots"""
    if state_col not in df.columns:
        print(f"  WARNING: {state_col} not in dataframe, skipping state-wise plots")
        return

    states = df[state_col].unique()
    n_states = len(states)

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

        feat_imp = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)

        feat_imp_top = feat_imp.head(top_k)

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
        print("  Computing permutation importance...")

        sample_size = min(500, len(X_test))
        sample_idx = np.random.RandomState(random_state).choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_idx]
        y_sample = y_test.iloc[sample_idx]

        perm_importance = permutation_importance(
            pipeline, X_sample, y_sample,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1
        )

        perm_imp = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        perm_imp_top = perm_imp.head(top_k)

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
    try:
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except:
        return list(X_train.columns)

def adaptive_feature_selection(df, target, corr_threshold=0.7, min_samples_for_selection=50):
    numeric_cols = df.select_dtypes(include=np.number).columns
    filtered_df = df.copy()

    print(f"Features before filtering: {len(df.columns)}")

    if target in numeric_cols and len(df) >= min_samples_for_selection:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)
        highcorrcols = corrs[abs(corrs) > corr_threshold].index.tolist()
        highcorrcols = [col for col in highcorrcols if col != target]
        print(f"Removing {len(highcorrcols)} correlated features (corr>{corr_threshold})")
        filtered_df = df.drop(columns=highcorrcols)
    else:
        print(f"Data has {len(df)} samples < min_samples_for_selection={min_samples_for_selection}. Skipping correlation filtering.")

    print(f"Features after filtering: {len(filtered_df.columns)}")
    return filtered_df


def adaptive_xgb_params(data_size):
    if data_size < 50:
        print("Using small data XGBoost parameters")
        return {
            'n_estimators': 10000,
            'learning_rate': 0.01,
            'max_depth': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 2,
            'reg_alpha': 4.0,
            'reg_lambda': 4.0,
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }
    else:
        print("Using large data XGBoost parameters")
        return {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }


def run_loocv(
    df, target, id_cols, mice_args, xgb_params, corr_threshold=0.7, permimportance_threshold=0.001, min_samples_for_selection=50,
    feature_selection_method='perm'  # 'perm', 'elasticnet', 'none'
):
    numeric_cols = df.select_dtypes(include=np.number).columns
    filtered_df = df.copy()
    if target in numeric_cols and len(df) >= min_samples_for_selection:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)
        highcorrcols = corrs[abs(corrs) > corr_threshold].index.tolist()
        highcorrcols = [col for col in highcorrcols if col != target]
        print(f"Removing {len(highcorrcols)} correlated features (corr>{corr_threshold})")
        filtered_df = df.drop(columns=highcorrcols)
    else:
        print(f"Skipping correlation filtering due to sample size {len(df)} < {min_samples_for_selection}")

    loo = LeaveOneOut()
    selected_feature_sets = []

    train_r2s, test_r2s = [], []
    train_adj_r2s, test_adj_r2s = [], []

    feature_names = None

    for fold_num, (train_idx, test_idx) in enumerate(loo.split(filtered_df), 1):
        df_train = filtered_df.iloc[train_idx].reset_index(drop=True)
        df_test = filtered_df.iloc[test_idx].reset_index(drop=True)

        df_train_imp = impute_dataframe(df_train, id_cols, [target], args=mice_args, progress=False, tag=f'loocv_{fold_num}_train')
        df_test_imp = impute_dataframe(df_test, id_cols, [target], args=mice_args, progress=False, tag=f'loocv_{fold_num}_test')

        drop_cols = [target] + [col for col in id_cols if col in df_train_imp.columns]
        X_train = df_train_imp.drop(columns=drop_cols)
        y_train = df_train_imp[target]
        X_test = df_test_imp.drop(columns=drop_cols)
        y_test = df_test_imp[target]

        preprocessor, *_ = build_preprocessor_from_X(X_train)
        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        if feature_names is None:
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = list(X_train.columns)

        if feature_selection_method == 'perm':
            model = XGBRegressor(**xgb_params)
            model.fit(X_train_t, y_train)
            perm_imp = permutation_importance(model, X_train_t, y_train, n_repeats=10, random_state=42, n_jobs=-1)
            selected_mask = perm_imp.importances_mean > permimportance_threshold

        elif feature_selection_method == 'elasticnet':
            from sklearn.linear_model import ElasticNetCV
            enet = ElasticNetCV(cv=3, l1_ratio=[0.1,0.5,0.9,1.0], n_jobs=-1,
                               random_state=42, max_iter=5000)
            enet.fit(X_train_t, y_train)
            selected_mask = enet.coef_ != 0

        else:
            # No feature selection, use all features
            selected_mask = np.array([True] * len(feature_names))

        selected_features = np.array(feature_names)[selected_mask]
        selected_feature_sets.append(selected_features)

    # Aggregate selected features across folds (union)
    all_selected_features = np.unique(np.concatenate(selected_feature_sets))
    print(f"Selected {len(all_selected_features)} features across {loo.get_n_splits(df)} LOOCV folds using '{feature_selection_method}'.")

    feature_indices = [i for i, f in enumerate(feature_names) if f in all_selected_features]

    cv_train_r2s, cv_test_r2s = [], []
    cv_train_adj_r2s, cv_test_adj_r2s = [], []

    for fold_num, (train_idx, test_idx) in enumerate(loo.split(filtered_df), 1):
        df_train = filtered_df.iloc[train_idx].reset_index(drop=True)
        df_test = filtered_df.iloc[test_idx].reset_index(drop=True)

        df_train_imp = impute_dataframe(df_train, id_cols, [target], args=mice_args, progress=False, tag=f'final_loocv_{fold_num}_train')
        df_test_imp = impute_dataframe(df_test, id_cols, [target], args=mice_args, progress=False, tag=f'final_loocv_{fold_num}_test')

        drop_cols = [target] + [col for col in id_cols if col in df_train_imp.columns]
        X_train = df_train_imp.drop(columns=drop_cols)
        y_train = df_train_imp[target]
        X_test = df_test_imp.drop(columns=drop_cols)
        y_test = df_test_imp[target]

        preprocessor, *_ = build_preprocessor_from_X(X_train)
        preprocessor.fit(X_train)
        X_train_t_full = preprocessor.transform(X_train)
        X_test_t_full = preprocessor.transform(X_test)

        X_train_t = X_train_t_full[:, feature_indices]
        X_test_t = X_test_t_full[:, feature_indices]

        model = XGBRegressor(**xgb_params)
        model.fit(X_train_t, y_train)

        train_r2 = model.score(X_train_t, y_train)
        test_r2 = model.score(X_test_t, y_test)
        cv_train_r2s.append(train_r2)
        cv_test_r2s.append(test_r2)
        cv_train_adj_r2s.append(adjusted_r2(train_r2, len(y_train), len(all_selected_features)))
        cv_test_adj_r2s.append(adjusted_r2(test_r2, len(y_test), len(all_selected_features)))

    print(f"Final LOOCV XGBoost results with selected features:")
    print(f"Train R2 mean ± std: {np.mean(cv_train_r2s):.4f} ± {np.std(cv_train_r2s):.4f}")
    print(f"Test R2 mean ± std: {np.mean(cv_test_r2s):.4f} ± {np.std(cv_test_r2s):.4f}")
    print(f"Train Adjusted R2 mean ± std: {np.mean(cv_train_adj_r2s):.4f} ± {np.std(cv_train_adj_r2s):.4f}")
    print(f"Test Adjusted R2 mean ± std: {np.mean(cv_test_adj_r2s):.4f} ± {np.std(cv_test_adj_r2s):.4f}")

    return all_selected_features, {
        'train_r2': cv_train_r2s,
        'test_r2': cv_test_r2s,
        'train_adj_r2': cv_train_adj_r2s,
        'test_adj_r2': cv_test_adj_r2s,
    }

def run_bootstrap(df, target, id_cols, mice_args, xgb_params, high_corr_threshold=0.70, n_iterations=100, random_state=42):
    """Run Bootstrap resampling for confidence intervals"""

    print(f"\n{'='*80}")
    print(f"Bootstrap Resampling ({n_iterations} iterations)")
    print(f"{'='*80}\n")

    # Remove highly correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target in numeric_cols:
        corrs = df[numeric_cols].corr()[target].sort_values(ascending=False)

        print("Top 10 features correlated with target:")
        for idx, (feat, corr_val) in enumerate(corrs.head(10).items(), 1):
            print(f"  {idx:2d}. {feat:<60s} {corr_val:>8.6f}")

        high_corr_cols = corrs[1:][(corrs[1:].abs() >= high_corr_threshold)].index.tolist()

        if high_corr_cols:
            print(f"\n⚠ Removing {len(high_corr_cols)} features:")
            for col in high_corr_cols:
                print(f"  • {col} (corr = {corrs[col]:.6f})")
            df = df.drop(columns=high_corr_cols)
            print(f"\nNew shape: {df.shape}\n")

    rmse_scores = []
    mae_scores = []

    print(f"Running {n_iterations} bootstrap iterations...")

    for i in range(n_iterations):
        # Bootstrap resample
        df_boot = resample(df, replace=True, n_samples=len(df), random_state=random_state + i)

        # Out-of-bag samples for testing
        oob_indices = list(set(df.index) - set(df_boot.index))
        if len(oob_indices) == 0:
            continue

        df_test = df.loc[oob_indices].reset_index(drop=True)
        df_train = df_boot.reset_index(drop=True)

        # MICE imputation
        df_train_imp = impute_dataframe(df_train, id_cols, [target], group_by_state=True, args=mice_args, progress=False, tag=f'bootstrap{i}_train')
        df_test_imp = impute_dataframe(df_test, id_cols, [target], group_by_state=True, args=mice_args, progress=False, tag=f'bootstrap{i}_test')

        cols_to_drop = [target] + [col for col in id_cols if col in df_train_imp.columns]

        X_train = df_train_imp.drop(columns=cols_to_drop)
        y_train = df_train_imp[target]
        X_test = df_test_imp.drop(columns=cols_to_drop)
        y_test = df_test_imp[target]

        # Preprocessing
        preprocessor, _, _, _, _ = build_preprocessor_from_X(X_train)
        preprocessor.fit(X_train)

        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # Train model
        model = XGBRegressor(**xgb_params)
        model.fit(X_train_t, y_train)

        # Predictions
        y_test_pred = model.predict(X_test_t)

        # Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        rmse_scores.append(test_rmse)
        mae_scores.append(test_mae)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations")

    # Calculate confidence intervals
    rmse_median = np.median(rmse_scores)
    rmse_ci_low = np.percentile(rmse_scores, 2.5)
    rmse_ci_high = np.percentile(rmse_scores, 97.5)

    mae_median = np.median(mae_scores)
    mae_ci_low = np.percentile(mae_scores, 2.5)
    mae_ci_high = np.percentile(mae_scores, 97.5)

    # Summary
    print(f"\n{'='*80}")
    print("Bootstrap summary (median and 95% CI)")
    print(f"{'='*80}\n")

    print(f"RMSE: {rmse_median:.4f} [{rmse_ci_low:.4f}, {rmse_ci_high:.4f}]")
    print(f"MAE:  {mae_median:.4f} [{mae_ci_low:.4f}, {mae_ci_high:.4f}]")

    print(f"\n{'='*80}\n")

    return {'rmse': rmse_scores, 'mae': mae_scores}

def main(data_path, target, id_cols, output_dir, cv_method='kfold', feature_selection_method='perm',
         n_cv_folds=5, n_bootstrap=100, cv_random_state=42, corr_threshold=0.7):
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} rows, {len(data.columns)} columns")

    mice_args = argparse.Namespace(
        mice_model='huber', mice_max_iter=1000, mice_tol=1e-4, reg_seed=42,
        huber_tol=1e-5, huber_epsilon=1.0, huber_alpha=0.5, huber_max_iter=1000,
        impute_targets='regression', allowleakage=False, imputecategorical=True,
        add_flags=True, nogroupbystate=False, noflags=False, nocatimpute=False,
        target=[target], progress=False
    )
    xgb_params = {
        'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 4,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 10,
        'reg_alpha': 2.0, 'reg_lambda': 2.0, 'tree_method': 'hist',
        'n_jobs': -1, 'random_state': 42, 'verbosity': 0
    }

    if cv_method == 'loocv':
        cv_results = run_loocv(
            data, target, id_cols, mice_args, xgb_params,
            corr_threshold=corr_threshold,
            permimportance_threshold=0.001,
            min_samples_for_selection=50,
            feature_selection_method=feature_selection_method
        )
    elif cv_method == 'bootstrap':
        # Implement or call your bootstrap function
        pass
    else:
        print("KFold not implemented; please use LOOCV or bootstrap")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(output_dir / f'{cv_method}_results.csv', index=False)
    print(f"Saved {cv_method} results CSV to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XGBoost with CV and feature selection")
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--id-cols', nargs='+', required=True)
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--cv-method', choices=['loocv', 'bootstrap'], default='loocv')
    parser.add_argument('--feature-selection-method', choices=['perm', 'elasticnet', 'none'], default='perm')
    parser.add_argument('--corr-threshold', type=float, default=0.7)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--n-bootstrap', type=int, default=100)
    parser.add_argument('--cv-random-state', type=int, default=42)

    args = parser.parse_args()

    main(args.data, args.target, args.id_cols, args.output_dir,
         args.cv_method, args.feature_selection_method,
         args.cv_folds, args.n_bootstrap, args.cv_random_state, args.corr_threshold)
