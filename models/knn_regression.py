# MIT License
# See LICENSE file in the project root or at https://opensource.org/license/mit
#
# Copyright (c) 2025 Landon Nguyen, Alex Nguyen, Robert Cocker
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import re
import sys
import time
import json
import joblib
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

"""
Example Usage: See Jupyter Notebooks for more information

python knn_regression.py --data ../data/Key_indicator_districtwise.csv --target Infant_Mortality_Rate_Imr_Total_Person --id-cols State_Name State_District_Name --correlation 60 --test-size 0.25 --random-state 42 --outdir knn

"""

warnings.filterwarnings("ignore")

sns.set_palette("husl")
plt.style.use('default')

logging.basicConfig(
    level=logging.DEBUG if '--debug' in sys.argv else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('knn_debug.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@contextmanager
def spinner_progress(total_steps=64):
    spinners = ['|', '/', '-', '\\']
    start = time.time()
    iter_count = 0
    
    def update(n_features):
        nonlocal iter_count
        iter_count += 1
        elapsed = time.time() - start
        eta = (elapsed/iter_count)*(total_steps-iter_count)/60 if iter_count>0 else 0
        pct = min(100, (iter_count/total_steps)*100)
        spinner = spinners[iter_count%4]
        sys.stdout.write(f'\r[{spinner} {iter_count:3d}/{total_steps}] {n_features:4d} feats | ETA: {eta:.0f}m | {pct:3.0f}% ')
        sys.stdout.flush()
    
    yield update
    elapsed = time.time() - start
    print(f'\r✅ RFECV Feature Selection Complete! {elapsed/60:.1f}m total')

def load_data(data_path):
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    raise ValueError('Unsupported file type. Use .csv or .parquet')

def print_dataset_stats(df, target_col):
    print("\n" + "="*80)
    print("📈 Raw Dataset Summary")
    print("="*80)
    
    total_samples, total_features = df.shape
    print(f"📊 Dataset Shape: ({total_samples}, {total_features})")
    print(f"🎯 Target Column: '{target_col}'")

    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / (total_samples * total_features)) * 100
    print(f"🔍 Total Missing Null/NaN Values: {total_missing:,} ({missing_pct:.2f}%)")

    target_missing = df[target_col].isnull().sum()
    target_missing_pct = (target_missing / total_samples) * 100
    print(f"🎯 Target Missing: {target_missing:,} / {total_samples} ({target_missing_pct:.1f}%)")
    
    if target_missing > 0:
        print(f"   ⚠️  Warning: Target has missing values!")

    feature_missing = df.drop(columns=[target_col]).isnull().sum()
    missing_features = feature_missing[feature_missing > 0].sort_values(ascending=False)
    
    if len(missing_features) > 0:
        print(f"\n📋 Top 10 Features with Missing Null/NaN Values:")
        print("-" * 60)
        for feature, count in missing_features.head(10).items():
            pct = (count / total_samples) * 100
            print(f"  {str(feature):40s} | {count:6,} ({pct:5.1f}%)")
        print(f"\n📊 Total features with missing values: {len(missing_features)}/{total_features - 1}")
    else:
        print("\n✅ No missing values in features!")
    
    dtype_counts = df.dtypes.value_counts()
    print(f"\n🔧 Data Types:")
    for dtype, count in dtype_counts.items():
        dtype_str = str(dtype)[:14] 
        print(f"  {dtype_str:15s} | {count:3d} columns")
    print("="*80)

def print_pre_rfecv_stats(X_processed, y_train, feature_names, num_features):
    print("\n" + "="*80)
    print("📈 Pre-RFECV Feature Selection Summary (Post-Preprocessing + Feature Correlation Drop)")
    print("="*80)
    
    n_samples, n_features = X_processed.shape
    print(f"📊 Processed Dataset: ({n_samples}, {n_features})")
    print(f"🎯 Target Samples: {len(y_train)}")
    print(f"📋 Feature Name Count: {len(feature_names)}")
    
    total_missing = np.isnan(X_processed).sum()
    missing_pct = (total_missing / (n_samples * n_features)) * 100
    print(f"🔍 Post-Preprocessing Missing Null/NaN Values: {total_missing:,} ({missing_pct:.2f}%)")
    
    if total_missing == 0:
        print("✅ No missing Null/NaN values after preprocessing!")
    else:
        print("⚠️  Warning: Missing Null/NaN values persist after preprocessing!")
    
    # Target stats
    y_train = np.asarray(y_train).ravel()
    target_missing = np.isnan(y_train).sum()
    print(f"🎯 Target Missing Null/NaN Values: {target_missing:,}/{len(y_train)} ({target_missing/len(y_train)*100:.1f}%)")
    
    numeric_feats = sum(1 for name in feature_names if any(c.isdigit() or c in '.-' for c in name))
    ohe_features = sum('__' in name for name in feature_names)
    print(f"\n🔧 Feature Breakdown:")
    print(f"  📊 Numerical Features: {numeric_feats}/{len(feature_names)}")
    print(f"  🅰️Categorical (Post-OHE): {len(feature_names) - numeric_feats}")
    print(f"  🔄 OHE Features Generated: {ohe_features}")

    if n_features > 0:
        corrs = np.corrcoef(X_processed.T, y_train)[-1, :-1]
        abs_corrs = np.abs(corrs)
        top5_idx = np.argsort(abs_corrs)[-5:][::-1]
        print(f"\n📊 Top 5 |corr| Features With Target Variable:")
        for i in top5_idx:
            print(f"  {str(feature_names[i])[:40]:40s} | {abs_corrs[i]:.4f}")
    print("="*80)

def find_target_columns(df, target_name):
    target_candidates = [col for col in df.columns if target_name.lower() in col.lower()]
    return target_candidates[0] if target_candidates else target_name.lower()

def find_state_columns(df, id_cols):
    candidates = []
    for col in id_cols:
        matching_cols = [c for c in df.columns if col.lower() in c.lower()]
        candidates.extend(matching_cols)
    state_patterns = ['state', 'region', 'province', 'county', 'district']
    for pattern in state_patterns:
        matches = df.columns[df.columns.str.lower().str.contains(pattern, na=False)]
        candidates.extend(matches.tolist())
    for col in candidates:
        if col in df.columns and df[col].nunique() > 1:
            return col
    return None

def calculate_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])

def get_feature_names(preprocessor):
    feature_names = list(preprocessor.get_feature_names_out())
    return [name.split('__', 1)[1] if '__' in name else name for name in feature_names]

def drop_feature_correlations(X_train, X_test, y_train, feature_names, drop_pct):
    drop_pct = float(max(0.0, min(100.0, drop_pct)))
    if drop_pct <= 0:
        return X_train, X_test, feature_names

    y_arr = np.asarray(y_train).ravel()
    n_features = X_train.shape[1]
    if n_features <= 1:
        return X_train, X_test, feature_names

    corrs = np.corrcoef(X_train.T, y_arr)[-1, :-1]
    abs_corrs = np.abs(corrs)

    n_drop = int(round(n_features * drop_pct/100.0))
    if n_drop <= 0 or n_drop >= n_features:
        return X_train, X_test, feature_names

    keep_idx = np.argsort(abs_corrs)[n_drop:]
    X_train_new = X_train[:, keep_idx]
    X_test_new = X_test[:, keep_idx]
    feature_names_new = [feature_names[i] for i in keep_idx]

    print(
        f"✓ Dropped {n_drop}/{n_features} features "
        f"({drop_pct:.2f}%) by |corr| with target; "
        f"{len(feature_names_new)} remain."
    )
    return X_train_new, X_test_new, feature_names_new

def plot_feature_target_correlations(X_processed, y_train, feature_names, selector_support, out_dir, top_n=25, chunk_size=5000):  
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X_processed)
    y = np.asarray(y_train).ravel()
    n_samples, n_features = X.shape
    if len(feature_names) != n_features:
        feature_names = feature_names[:n_features]
    if len(selector_support) != n_features:
        selector_support = selector_support[:n_features]

    top_n = min(top_n, n_features)
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        print("Target has zero variance; skipping correlation plot.")
        return

    best = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        X_chunk = X[:, start:end]    

        Xm = X_chunk - X_chunk.mean(axis=0, keepdims=True)
        ym = y - y_mean

        num = (Xm * ym[:, None]).sum(axis=0) 
        std_x = Xm.std(axis=0, ddof=0)
        denom = std_x * y_std * n_samples
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_chunk = np.where(denom != 0, num / denom, 0.0)

        abs_corr_chunk = np.abs(corr_chunk)
        for local_i, (ac, c) in enumerate(zip(abs_corr_chunk, corr_chunk)):
            g_i = start + local_i
            if len(best) < top_n:
                best.append((ac, c, g_i))
                if len(best) == top_n:
                    best.sort(key=lambda x: x[0])
            else:
                if ac > best[0][0]:
                    best[0] = (ac, c, g_i)
                    best.sort(key=lambda x: x[0])
    if not best:
        print("No correlations computed; skipping correlation plot.")
        return

    best.sort(key=lambda x: x[0], reverse=True)
    top_corrs = np.array([b[1] for b in best])
    top_idx = np.array([b[2] for b in best], dtype=int)
    top_features = [feature_names[i] for i in top_idx]
    is_selected = [bool(selector_support[i]) for i in top_idx]
    colors = [
        'steelblue' if selected else ('coral' if corr > 0 else 'darkgreen')
        for corr, selected in zip(top_corrs, is_selected)
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])
    bars1 = ax1.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7, height=0.7)
    ax1.set_yticks(range(len(top_corrs)))
    ax1.set_yticklabels(
        [f[:35] + '...' if len(f) > 35 else f for f in top_features],
        fontsize=10
    )
    ax1.set_xlabel('Correlation with Target', fontsize=12, weight='bold')
    ax1.set_title('Top KNN Feature-Target Correlations Selected', fontsize=14, weight='bold')
    ax1.grid(axis='x', alpha=0.5)
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
    for bar, corr in zip(bars1, top_corrs):
        width = bar.get_width()
        ax1.text(
            width + (0.01 if width >= 0 else -0.03),
            bar.get_y() + bar.get_height() / 2,
            f'{corr:.3f}',
            ha='left' if width >= 0 else 'right',
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    selected_count = sum(is_selected)
    ax1.text(
        0.02, 0.98,
        f'RFECV Selected: {selected_count}/{len(top_corrs)}',
        transform=ax1.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
        fontsize=11
    )

    selected_corrs = [c for c, s in zip(top_corrs, is_selected) if s]
    non_selected_corrs = [c for c, s in zip(top_corrs, is_selected) if not s]
    ax2.hist(
        [selected_corrs, non_selected_corrs],
        bins=10,
        alpha=0.7,
        label=['Selected', 'Not Selected'],
        color=['steelblue', 'coral'],
        density=True
    )
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_ylabel('Density')
    ax2.set_title('Correlation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(out_dir)/'feature_target_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ feature_target_correlations.png")

def plot_true_vs_pred(y_true, y_pred, out_dir, subset_label, metrics):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, color='steelblue', label='True', s=35, alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, color='coral', label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'{subset_label} Predictions')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    textstr = f'R²: {metrics["R2"]:.3f}\nAdj R²: {metrics["Adj_R2"]:.3f}\nRMSE: {metrics["RMSE"]:.3f}\nMAE: {metrics["MAE"]:.3f}'
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{subset_label.lower()}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {subset_label.lower()}_scatter.png")

def plot_residuals(y_true, y_pred, out_dir, split_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].scatter(y_pred, residuals, alpha=0.7, s=40, color='steelblue')
    axes[0,0].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    axes[0,0].grid(True, alpha=0.5)
    
    axes[0,1].hist(residuals, bins=10, color='steelblue', alpha=0.7, edgecolor='coral')
    axes[0,1].set_xlabel('Residuals')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Residuals Distribution')
    axes[0,1].grid(True, alpha=0.5)
    
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1,0].get_lines()[0].set_markeredgecolor('coral')
    axes[1,0].set_title('Q-Q Plot (Normality)')
    
    axes[1,1].scatter(range(len(residuals)), residuals, alpha=0.7, s=20, color='steelblue')
    axes[1,1].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Index')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Index')
    axes[1,1].grid(True, alpha=0.5)
    
    plt.suptitle(f'{split_name} Residuals Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f'{split_name.lower()}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {split_name.lower()}_residuals.png")

def plot_residuals_granularity(y_true, y_pred, outdir, split_name, jitter_level=1e-6):
    residuals = y_true - y_pred
    residuals = np.asarray(residuals).ravel()
    jitter = np.random.normal(0, jitter_level, size=residuals.shape)
    residuals_jitter = residuals + jitter

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals_jitter, alpha=0.7, s=40, color="steelblue")
    plt.axhline(0, color="coral", linestyle="--", linewidth=2)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"{split_name} Residuals Plot (Granularity)")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"{split_name.lower()}_residuals_granularity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {split_name.lower()}_residuals_granularity.png")

def plot_feature_importance(importances, feature_names, selector_support, out_dir, top_n=25):
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:][::-1]
    colors = ['steelblue' if selector_support[i] else 'coral' for i in idx]
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(top_n), importances[idx], color=colors, alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i][:35] + '...' if len(feature_names[i]) > 35 else feature_names[i] for i in idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest RFECV Feature Importances (Inputs to KNN Model)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ feature_importance.png")

def plot_feature_target_correlations(X_processed, y_train, feature_names, selector_support, out_dir, top_n=25, chunk_size=5000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X_processed)
    y = np.asarray(y_train).ravel()
    n_samples, n_features = X.shape
    if len(feature_names) != n_features:
        feature_names = feature_names[:n_features]
    if len(selector_support) != n_features:
        selector_support = selector_support[:n_features]

    top_n = min(top_n, n_features)
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        print("Target has zero variance; skipping correlation plot.")
        return
        
    best = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        X_chunk = X[:, start:end]

        Xm = X_chunk - X_chunk.mean(axis=0, keepdims=True)
        ym = y - y_mean

        num = (Xm * ym[:, None]).sum(axis=0)
        std_x = Xm.std(axis=0, ddof=0)
        denom = std_x * y_std * n_samples
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_chunk = np.where(denom != 0, num / denom, 0.0)

        abs_corr_chunk = np.abs(corr_chunk)
        for local_i, (ac, c) in enumerate(zip(abs_corr_chunk, corr_chunk)):
            g_i = start + local_i
            if len(best) < top_n:
                best.append((ac, c, g_i))
                if len(best) == top_n:
                    best.sort(key=lambda x: x[0])
            else:
                if ac > best[0][0]:
                    best[0] = (ac, c, g_i)
                    best.sort(key=lambda x: x[0])

    if not best:
        print("No correlations computed; skipping correlation plot.")
        return

    best.sort(key=lambda x: x[0], reverse=True)
    top_corrs = np.array([b[1] for b in best])
    top_idx = np.array([b[2] for b in best], dtype=int)
    top_features = [feature_names[i] for i in top_idx]
    is_selected = [bool(selector_support[i]) for i in top_idx]

    colors = ["steelblue" if sel else ("coral" if corr > 0 else "darkgreen") for corr, sel in zip(top_corrs, is_selected)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])
    bars1 = ax1.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7, height=0.7)
    ax1.set_yticks(range(len(top_corrs)))
    ax1.set_yticklabels([f[:35] + "..." if len(f) > 35 else f for f in top_features], fontsize=10)
    ax1.set_xlabel("Correlation with Target", fontsize=12, weight="bold")
    ax1.set_title("Top KNN Feature-Target Correlations Selected", fontsize=14, weight="bold")
    ax1.grid(axis="x", alpha=0.5)
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5)

    for bar, corr in zip(bars1, top_corrs):
        width = bar.get_width()
        ax1.text(
            width + (0.01 if width >= 0 else -0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{corr:.3f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    selected_count = sum(is_selected)
    ax1.text(
        0.02,
        0.98,
        f"RFECV Selected: {selected_count}/{len(top_corrs)}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
        fontsize=11,
    )

    selected_corrs = [c for c, s in zip(top_corrs, is_selected) if s]
    non_selected_corrs = [c for c, s in zip(top_corrs, is_selected) if not s]
    ax2.hist(
        [selected_corrs, non_selected_corrs],
        bins=10,
        alpha=0.7,
        label=["Selected", "Not Selected"],
        color=["steelblue", "coral"],
        density=True,
    )
    ax2.set_xlabel("Correlation Coefficient")
    ax2.set_ylabel("Density")
    ax2.set_title("Correlation Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "feature_target_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ feature_target_correlations.png")

def plot_statewise_histogram(df, valuecol, statecol, outdir):
    plt.figure(figsize=(14, 8))
    state_means = df.groupby(statecol)[valuecol].mean().sort_values()
    states_order = state_means.index.tolist()
    plot_data = df[df[statecol].isin(states_order)]
    palette = sns.color_palette("husl", n_colors=len(states_order))

    sns.histplot(
        data=plot_data,
        x=valuecol,
        hue=statecol,
        hue_order=states_order,
        element="step",
        stat="count",
        common_norm=False,
        palette=palette,
        alpha=0.4,
        multiple="layer", 
    )

    plt.title("Infant Mortality Rate by State")
    plt.xlabel(valuecol)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    plt.savefig(Path(outdir)/"statewise_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ statewise_histogram.png")

def plot_statewise_facets(df, value_col, state_col, out_dir):
    top_states = df[state_col].value_counts().head(9).index
    plot_data = df[df[state_col].isin(top_states)]

    g = sns.FacetGrid(plot_data, col=state_col, col_wrap=3, height=3, aspect=1.3, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x=value_col, bins=9, color="steelblue", alpha=0.7)
    
    for ax in g.axes.flatten():
        if ax is not None:
            ax.grid(True, alpha=0.5)

    g.set_titles("{col_name}")
    g.set_axis_labels(value_col, "Count")
    g.fig.suptitle(f"{value_col} by {state_col}", y=1.02)
    plt.savefig(Path(out_dir) / "statewise_facets.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ statewise_facets.png")

def plot_prediction_distribution(y_true, y_pred, out_dir, split_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    bins = 7

    plt.hist(
        y_true,
        bins=bins,
        histtype="step",
        linewidth=2,
        color="steelblue",
        alpha=0.9,
        density=True,
        label="True",
    )

    plt.hist(
        y_pred,
        bins=bins,
        histtype="step",
        linewidth=2,
        color="coral",
        alpha=0.9,
        density=True,
        label="Predicted",
    )

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"{split_name} Distribution")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.7, s=40, color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="coral", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{split_name.lower()}_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ {split_name.lower()}_distribution.png")

def plot_model_comparison(train_metrics, test_metrics, out_dir):
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    x = np.arange(len(metrics_df))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, metrics_df['R2'], width, label='R²', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, metrics_df['Adj_R2'], width, label='Adj R²', color='coral', alpha=0.7)
    ax.set_xlabel('Split')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Train vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ model_comparison.png")

def print_outlier_analysis(y_true, y_pred, split_name, out_dir, df_deduped=None, orig_indices=None, state_col='State_Name', district_col='State_District_Name'):
    print(f"\n🔍 {split_name} Outlier Analysis")
    print("-"*80)
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    
    Q1, Q3 = np.percentile(residuals, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    stat_outliers = np.abs(residuals) > np.max(np.abs([lower_bound, upper_bound]))
    stat_outlier_pct = stat_outliers.sum() / len(residuals) * 100
    
    print(f"📊 Statistical Outliers (IQR 1.5): {stat_outliers.sum():3d}/{len(residuals):3d} ({stat_outlier_pct:4.1f}%)")
    
    worst_idx = np.argsort(abs_residuals)[-5:][::-1]
    print(f"\n🚨 Top 5 KNN Regression Model Inference Worst Predictions (Residuals Errors):")
    print(f"{'#':3s} | {'State':15s} | {'District':20s} | {'True':6s} | {'Pred':6s} | {'Error':6s}")
    print("-"*80)
    
    for i, rel_idx in enumerate(worst_idx):
        orig_idx = int(orig_indices[rel_idx]) if orig_indices is not None else rel_idx
        
        state = "N/A"
        district = "N/A"
        if df_deduped is not None:
            try:
                if 0 <= orig_idx < len(df_deduped):
                    if state_col in df_deduped.columns:
                        state = str(df_deduped.iloc[orig_idx].get(state_col, "N/A"))[:14]
                    if district_col in df_deduped.columns:
                        district = str(df_deduped.iloc[orig_idx].get(district_col, "N/A"))[:19]
            except Exception:
                state, district = "LookupError", "LookupError"
        
        print(f"{i+1:2d}  | {state:15s} | {district:20s} | "
              f"{y_true[rel_idx]:6.1f} | {y_pred[rel_idx]:6.1f} | {abs_residuals[rel_idx]:6.1f}")
    
    outliers_df = pd.DataFrame({
        'relative_idx': np.arange(len(residuals)),
        'original_index': orig_indices if orig_indices is not None else np.arange(len(residuals)),
        'true': y_true, 'pred': y_pred, 'residual': residuals,
        'is_outlier': stat_outliers,
        'abs_error': abs_residuals
    })
    
    if df_deduped is not None:
        states = []
        districts = []
        for i, orig_idx in enumerate(outliers_df['original_index']):
            try:
                idx = int(orig_idx)
                if 0 <= idx < len(df_deduped):
                    states.append(str(df_deduped.iloc[idx].get(state_col, "N/A")))
                    districts.append(str(df_deduped.iloc[idx].get(district_col, "N/A")))
                else:
                    states.append("OutOfBounds")
                    districts.append("OutOfBounds")
            except:
                states.append("Error")
                districts.append("Error")
        outliers_df['state'] = states
        outliers_df['district'] = districts
    
    outliers_df.to_csv(Path(out_dir)/f'{split_name.lower()}_outliers_summary.csv', index=False)
    print(f"💾 Saved: {split_name.lower()}_outliers_summary.csv")

def save_metrics(train_metrics, test_metrics, out_dir):
    results = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    results.index.name = 'Split'
    results.to_csv(Path(out_dir)/'metrics.csv', float_format='%.4f')
    print("✓ metrics.csv")

def main(args):
    print("KNN Regression Model")
    print("="*70)
    df = load_data(args.data)
    original_target = args.target
    df.columns = [re.sub(r'^([A-Z]{2})_', '', col) for col in df.columns]
    args.target = find_target_columns(df, original_target)
    
    print(f"📊 Dataset: {df.shape}")
    print(f"🎯 Target: '{original_target}' → '{args.target}'")

    id_cols_fixed = [re.sub(r'^([A-Z]{2})_', '', col) for col in args.id_cols]
    
    print(f"🧹 Deduplicating: {len(df)} → ", end="")
    mask = ~df.duplicated().values
    df_deduped = df[mask].copy()
    print(f"{len(df_deduped)} samples")
    print_dataset_stats(df_deduped, args.target)

    if df_deduped[args.target].isnull().any():
        print("❌ ERROR: Target has missing values. Dropping incomplete rows.")
        df_deduped = df_deduped.dropna(subset=[args.target])
        print(f"📊 After target cleanup: {df_deduped.shape}")
    print(f"✅ Raw dataset for preprocessing ({df_deduped.shape})")
    
    if args.target not in df_deduped.columns:
        raise ValueError(f"Target '{args.target}' not found")
    
    X = df_deduped.drop(columns=[args.target] + [col for col in id_cols_fixed if col in df_deduped.columns])
    y = df_deduped[args.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True)

    train_indices = X_train.index.values
    test_indices = X_test.index.values
    
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    state_col = find_state_columns(df_deduped, id_cols_fixed)
    print(f"🗺️ State Column: {state_col or 'None'}")
    if state_col and state_col in df_deduped.columns:
        plot_statewise_histogram(df_deduped, args.target, state_col, out_dir)
        plot_statewise_facets(df_deduped, args.target, state_col, out_dir)
    
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    X_train_processed, X_test_processed, feature_names = drop_feature_correlations(
    X_train_processed, X_test_processed, y_train, feature_names, args.correlation)


    num_features = X_train_processed.shape[1]
    target_min, target_max = y.min(), y.max()
    print(f"✅ Features: {num_features} | Target Range: {target_min:.1f}-{target_max:.1f}")
    print_pre_rfecv_stats(X_train_processed, y_train, feature_names, num_features)
    print("🔍 RFECV Feature Selector:") 
    print("🔍 Running a pass on the data..")
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    with spinner_progress(min(64, X_train_processed.shape[1]//10)):
        selector = RFECV(rf, step=10, cv=cv, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
        print("🔍 X_train Shape:", X_train.shape)
        print("🔍 X_train_processed Shape:", X_train_processed.shape)
        print("🔍 First 5 Original Features:", list(X_train.columns[:5]))
        print(f"✅ RFECV Input: {X_train_processed.shape[1]} features")
        print(f"✅ Original Numeric Columns: {len(X_train.select_dtypes(include=np.number).columns)} features")
        selector.fit(X_train_processed, y_train)
        print(f"✅ RFECV: {X_train_processed.shape[1]} → {selector.n_features_} features selected")
    
    X_train_selected = selector.transform(X_train_processed)
    X_test_selected = selector.transform(X_test_processed)
    
    model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
    model.fit(X_train_selected, y_train)
    
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    
    n_train, p = len(y_train), X_train_selected.shape[1]
    n_test = len(y_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = calculate_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_metrics = {'R2': train_r2, 'Adj_R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj_R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    
    joblib.dump(model, out_dir/'knn_model.joblib')
    joblib.dump(preprocessor, out_dir/'preprocessor.joblib')
    joblib.dump(selector, out_dir/'rfecv_selector.joblib')
    save_metrics(train_metrics, test_metrics, out_dir)
    
    print("\n📈 Generating model inference plots...")
    plot_residuals(y_train, y_train_pred, out_dir, "Train")
    plot_residuals(y_test, y_test_pred, out_dir, "Test")
    plot_residuals_granularity(y_train, y_train_pred, out_dir, "Training")
    plot_residuals_granularity(y_test, y_test_pred, out_dir, "Testing")
    plot_feature_target_correlations(X_train_processed, y_train, feature_names, selector.support_, out_dir)
    plot_true_vs_pred(y_train, y_train_pred, out_dir, 'Train', train_metrics)
    plot_true_vs_pred(y_test, y_test_pred, out_dir, 'Test', test_metrics)
    plot_residuals(y_train, y_train_pred, out_dir, 'Train')
    plot_residuals(y_test, y_test_pred, out_dir, 'Test')
    plot_feature_importance(selector.estimator_.feature_importances_, feature_names, selector.support_, out_dir)
    plot_prediction_distribution(y_train, y_train_pred, out_dir, 'Train')
    plot_prediction_distribution(y_test, y_test_pred, out_dir, 'Test')
    plot_model_comparison(train_metrics, test_metrics, out_dir)
    print_outlier_analysis(y_train, y_train_pred, "Train", out_dir, df_deduped=df_deduped, orig_indices=train_indices, 
                           state_col='State_Name', district_col='State_District_Name')
    print_outlier_analysis(y_test, y_test_pred, "Test", out_dir, df_deduped=df_deduped, orig_indices=test_indices, 
                           state_col='State_Name', district_col='State_District_Name')
    print("\n" + "="*70)
    print("✅ KNN Model Results:")
    print("="*70)
    print(f"🎯 Test: R²={test_r2:.4f} | Adj R²={test_adj_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
    print(f"📊 Features: {selector.n_features_}/{len(feature_names)}")
    print(f"📁 Outputs: {out_dir}")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Regression w/ 13 Color-Matched Plots')
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--id-cols', nargs='+', default=[])
    parser.add_argument("--correlation", type=float, default=0.0, help="Drop features by correlation % before feature selection")
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--outdir', default='artifacts')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
