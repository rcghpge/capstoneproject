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
import json
import joblib
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
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
Example Usage: See Jupyter Notebooks for more information.

!python knn_base.py \
--data ../data/Key_indicator_districtwise.csv \
--target YY_Infant_Mortality_Rate_Imr_Total_Person \
--id-cols State_Name State_District_Name \
--test-size 0.25 --random-state 42 --outdir knn

"""

warnings.filterwarnings("ignore")
sns.set_palette("husl")
plt.style.use('default')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(datapath):
    df = pd.read_csv(datapath)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def verify_columns(df, target, id_cols):
    print("\n🔍 Verifying Features:")
    print(f"✓  Target: '{target}' → {'✅ Found' if target in df.columns else '❌ Missing'}")
    
    for id_col in id_cols:
        status = '✅ Found' if id_col in df.columns else '❌ Missing'
        print(f"✓  ID: '{id_col}' → {status}")
    print(f"✓  Dropping selected features from the training data..")
    if target not in df.columns:
        print("\n📋 All columns with 'Infant'/'IMR':")
        for col in df.columns:
            if 'infant' in col.lower() or 'imr' in col.lower():
                print(f"   - '{col}'")
        raise KeyError(f"Target '{target}' not found!")

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def get_raw_feature_names(preprocessor, original_cols):
    feat_names = list(preprocessor.get_feature_names_out())
    raw_mapping = {}
    
    for feat in feat_names:
        if feat.startswith('num__'):
            raw_mapping[feat] = feat[5:]
        elif feat.startswith('cat__'):
            raw_mapping[feat] = feat[5:].split('_', 1)[1] if '_' in feat[5:] else feat[5:]
    
    return [raw_mapping.get(name, name) for name in feat_names]

def plot_true_vs_pred(y_true, y_pred, outdir, subset_label, metrics):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, color='steelblue', label='True', s=40, alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, color='coral', label='Predicted', linewidth=3, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'{subset_label} Predictions')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    text_str = (f'R²: {metrics["R2"]:.3f}\n'
                f'R_adj: {metrics["Adj R2"]:.3f}\n'
                f'RMSE: {metrics["RMSE"]:.3f}\n'
                f'MAE: {metrics["MAE"]:.3f}')
    plt.gca().text(0.02, 0.98, text_str, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{subset_label.lower()}_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {subset_label.lower()}_scatter.png")

def plot_residuals(y_true, y_pred, outdir, split_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].scatter(y_pred, residuals, alpha=0.7, s=40, color='steelblue')
    axes[0,0].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].hist(residuals, bins=10, color='steelblue', alpha=0.7, edgecolor='coral')
    axes[0,1].set_xlabel('Residuals')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Residuals Distribution')
    axes[0,1].grid(True, alpha=0.5)
    
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1,0].get_lines()[0].set_markeredgecolor('coral')
    axes[1,0].set_title('Q-Q Plot (Normality)')
    
    axes[1,1].scatter(range(len(residuals)), residuals, alpha=0.6, s=20, color='steelblue')
    axes[1,1].axhline(0, color='coral', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Index')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Index')
    axes[1,1].grid(True, alpha=0.5)
    
    plt.suptitle(f'{split_name} Residuals Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{split_name.lower()}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {split_name.lower()}_residuals.png")

def plot_feature_importance(importances, feature_names, selector_support, outdir, top_n=20):
    def clean_name(name: str) -> str:
        for prefix in ("num__", "cat__"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        # remove your domain prefix
        if name.startswith("AA_"):
            name = name[3:]
        return name

    clean_feature_names = [clean_name(n) for n in feature_names]

    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[-top_n:][::-1]
    colors = ["darkgreen" if selector_support[i] else "steelblue" for i in idx]

    plt.figure(figsize=(12, 10))
    plt.barh(range(top_n), importances[idx], color=colors, alpha=0.7)

    labels = []
    for i in idx:
        n = clean_feature_names[i]
        labels.append(n[:35] + "..." if len(n) > 35 else n)

    plt.yticks(range(top_n), labels)
    plt.xlabel("Feature Importance")
    plt.title("Model Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(outdir)/"feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: feature_importance.png")

def plot_prediction_distribution(y_true, y_pred, outdir, split_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_true, bins=7, alpha=0.6, label='True', color='steelblue', density=True)
    plt.hist(y_pred, bins=7, alpha=0.6, label='Predicted', color='coral', density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{split_name} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6, s=40, color='steelblue')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'coral', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{split_name.lower()}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {split_name.lower()}_distribution.png")

def plot_model_comparison(train_metrics, test_metrics, outdir):
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    x = np.arange(len(metrics_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, metrics_df['R2'], width, label='R²', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, metrics_df['Adj R2'], width, label='Adj R²', color='coral', alpha=0.8)
    ax.set_xlabel('Split')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Train vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend()
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(Path(outdir)/'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_comparisons.png")

def plot_jitter_true_vs_pred(y_true, y_pred, outdir, split_name, jitter_level=1e-6):
    x_jitter = np.random.normal(0, jitter_level, len(y_true))
    y_jitter = np.random.normal(0, jitter_level, len(y_pred))
    
    x_jittered = y_true + x_jitter
    y_jittered = y_pred + y_jitter
    
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(x_jittered, y_jittered, alpha=0.7, s=80, c=y_pred-y_true, 
                         cmap='RdYlBu_r', edgecolors='black', linewidth=0.8)
    
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'coral', lw=3, alpha=0.9, label='Perfect Prediction')
    plt.axvline(y_true.mean(), color='steelblue', linestyle='--', alpha=0.7, label=f'True Mean: {y_true.mean():.1f}')
    plt.axhline(y_pred.mean(), color='coral', linestyle='--', alpha=0.7, label=f'Pred Mean: {y_pred.mean():.1f}')
    plt.xlabel('True Values (Jittered)')
    plt.ylabel('Predicted Values (Jittered)')
    plt.title(f'{split_name} Jittered True vs Predicted\n(Jitter={jitter_level}, Color=Residual)')
    plt.colorbar(scatter, label='Prediction Error (Pred-True)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    textstr = (f'R²: {r2_score(y_true, y_pred):.3f}\n'
               f'RMSE: {rmse:.3f}\n'
               f'MAE: {mae:.3f}\n'
               f'N: {len(y_true):,}')
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(Path(outdir)/f'{split_name.lower()}_jitter_true_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {split_name.lower()}_jitter_true_vs_pred.png")

def print_selected_features_raw(selector, raw_feature_names, top_n=20):
    print("\n" + "="*80)
    print("🎯 RFECV Top Selected Features - Raw Feature Names (RF Regressor):")
    print("="*80)
    
    selected_mask = selector.support_
    selected_raw = [raw_feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    
    for i, feat in enumerate(selected_raw[:top_n]):
        print(f"  {i+1:2d}. '{feat}'")
    
    total = len(selected_raw)
    print(f"\n📊 Selected: {total}/{len(raw_feature_names)} ({100*total/len(raw_feature_names):.1f}%)")
    print("="*80)

def calculate_adjusted_r2(r2_score, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2_score) * (n_samples - 1)/(n_samples - n_features - 1)

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    print("KNN Regression Model")
    print(f"📁 Output: {outdir}")
    print("="*80)
    
    df = load_data(args.data)
    verify_columns(df, args.target, args.id_cols)
    id_cols_found = [col for col in args.id_cols if col in df.columns]
    X = df.drop(columns=[args.target] + id_cols_found)
    y = df[args.target]
    
    print(f"✅ Features: {X.shape[1]} | Target range: {y.min():.1f}-{y.max():.1f}")
    
    mask = ~df.duplicated(subset=X.columns.tolist())
    df_deduped = df[mask]
    X = df_deduped.drop(columns=[args.target] + id_cols_found)
    y = df_deduped[args.target]
    print(f"✅ Deduplicated: {len(X):,} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    raw_feature_names = get_raw_feature_names(preprocessor, X_train.columns.tolist())
    
    print(f"✅ Processed: {X_train_proc.shape}")
    print("\n🎯 RFECV Feature Selection... Running a pass on the data")
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    selector = RFECV(rf, step=0.1, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    selector.fit(X_train_proc, y_train)
    
    print(f"✅ Selected: {selector.n_features_} features")
    print_selected_features_raw(selector, raw_feature_names)
    
    X_train_sel = selector.transform(X_train_proc)
    X_test_sel = selector.transform(X_test_proc)
    
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='manhattan')
    knn.fit(X_train_sel, y_train)
    
    y_train_pred = knn.predict(X_train_sel)
    y_test_pred = knn.predict(X_test_sel)
    
    n_train, p = len(y_train), X_train_sel.shape[1]
    n_test = len(y_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_adj_r2 = calculate_adjusted_r2(train_r2, n_train, p)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_metrics = {'R2': train_r2, 'Adj R2': train_adj_r2, 'RMSE': train_rmse, 'MAE': train_mae}
    test_metrics = {'R2': test_r2, 'Adj R2': test_adj_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    
    print(f"\n📊 KNN Regression Model Inference Metrics:")
    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
    
    joblib.dump(knn, outdir/'knn_model.joblib')
    joblib.dump(preprocessor, outdir/'preprocessor.joblib')
    joblib.dump(selector, outdir/'rfecv_selector.joblib')
    
    print("\n📈 Generating KNN model inference plots...")
    print("-" * 50)

    raw_feature_names = get_raw_feature_names(preprocessor, X_train.columns.tolist())
    plot_true_vs_pred(y_train, y_train_pred, outdir, 'Train', train_metrics)
    plot_true_vs_pred(y_test, y_test_pred, outdir, 'Test', test_metrics)
    plot_residuals(y_train, y_train_pred, outdir, 'Train')
    plot_residuals(y_test, y_test_pred, outdir, 'Test')
    plot_feature_importance(selector.estimator_.feature_importances_, raw_feature_names, selector.support_, outdir)
    plot_prediction_distribution(y_train, y_train_pred, outdir, 'Train')
    plot_prediction_distribution(y_test, y_test_pred, outdir, 'Test')
    plot_model_comparison(train_metrics, test_metrics, outdir)
    
    plot_jitter_true_vs_pred(y_train, y_train_pred, outdir, 'Train', jitter_level=0.3)
    plot_jitter_true_vs_pred(y_test, y_test_pred, outdir, 'Test', jitter_level=0.3)
    
    selected_features = [raw_feature_names[i] for i, sel in enumerate(selector.support_) if sel]
    pd.Series(selected_features).to_csv(outdir/'selected_features_raw.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ Generated Results:")
    print("="*70)
    print(f"📁 All files saved to: {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN regression model with RFECV feature selection for IMR assessment")
    parser.add_argument("--data", required=True, help="CSV file")
    parser.add_argument("--target", required=True, help="Exact target column name")
    parser.add_argument("--id-cols", nargs="*", default=[], help="ID columns")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--outdir", default="artifacts/knn")
    args = parser.parse_args()
    main(args)
