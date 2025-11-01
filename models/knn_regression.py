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
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
"""
Example Usage:
python -m models.knn_regression --data data/Key_indicator_districtwise.csv \
--target Infant_Mortality_Rate_Imr_Total_Person \
--id-cols State_Name State_District_Name \
--test-size 0.25 \
--random-state 42 \
--outdir knn
"""

def load_data(data_path):
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError('Unsupported file type')

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def plot_and_save(y_true, y_pred, out_dir):
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.savefig(Path(out_dir) / 'predicted_vs_actual.png')
    plt.close()

    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--', color='r')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.savefig(Path(out_dir) / 'residuals_vs_fitted.png')
    plt.close()

    plt.figure()
    plt.hist(residuals, bins=30, alpha=0.8)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    plt.savefig(Path(out_dir) / 'residual_histogram.png')
    plt.close()

def plot_true_vs_pred(y_true, y_pred, out_dir, subset_label, metrics):
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y_true)), y_true, color='black', label='True Value', s=20)
    plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Value')
    plt.xlabel('District')
    plt.ylabel('Infant Mortality Rate')
    plt.title(f'Baseline {subset_label} Scatter Plot')
    plt.legend()

    textstr = '\n'.join((
        f"R^2: {metrics['R2']:.2f}",
        f"Adjusted R^2: {metrics.get('Adjusted R2', 0):.2f}",
        f"RMSE: {metrics['RMSE']:.2f}",
        f"MAE: {metrics.get('MAE', 0):.2f}",
    ))
    plt.gca().text(0.95, 0.1, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f'baseline_{subset_label.lower()}_scatter.png')
    plt.close()

def get_feature_names(preprocessor):
    num_features = preprocessor.named_transformers_['num'].named_steps['imputer'].feature_names_in_
    cat_pipeline = preprocessor.named_transformers_['cat']
    cat_cols = preprocessor.transformers_[1][2]
    cat_features = []
    for i, cat in enumerate(cat_cols):
        cats = cat_pipeline.named_steps['onehot'].categories_[i]
        cat_features.extend([f"{cat}_{c}" for c in cats])
    return list(num_features) + cat_features

def plot_feature_importances(importances, feature_names, out_dir, top_n=10):
    top_n = min(top_n, len(importances), len(feature_names))
    idx = np.argsort(importances)[-top_n:]
    top_importances = importances[idx]
    top_features = np.array(feature_names)[idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_n), top_importances, align='center', color='steelblue')
    plt.yticks(range(top_n), top_features)
    plt.xlabel("Feature Importances")
    plt.title(f"Top {top_n} KNN Model Features")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'feature_importances.png')
    plt.close()

def plot_statewise_histogram(df, value_col, state_col, out_dir):
    plt.figure()
    sns.histplot(data=df, x=value_col, hue=state_col, element="step", stat="count", common_norm=False, palette='bright', alpha=0.4)
    plt.title('Infant Mortality Rate State-Wise')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'statewise_histogram.png')
    plt.close()

def plot_statewise_facets(df, value_col, state_col, out_dir):
    import seaborn as sns
    g = sns.FacetGrid(df, col=state_col, col_wrap=3, height=3, aspect=1.5, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x=value_col, stat="count", bins=10, color='steelblue')
    g.set_titles("{col_name}")
    g.set_axis_labels("Infant Mortality Rate", "Count")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Infant Mortality Rate State-Wise')
    plt.savefig(Path(out_dir) / 'statewise_facets.png')
    plt.close()

def save_metrics_table(train_metrics, test_metrics, out_dir):
    import pandas as pd
    results = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    results.index.name = 'Subset'
    results.to_csv(Path(out_dir) / "regression_metrics_table.csv")
    try:
        import dataframe_image as dfi
        dfi.export(results, Path(out_dir) / "regression_metrics_table.png")
    except:
        pass

def save_metrics(metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / 'metrics.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')

def main(args):
    df = load_data(args.data)
    #df.columns = [re.sub(r'^[A-Z]{2}_', '', col) for col in df.columns]
    #gid_cols_fixed = [re.sub(r'^[A-Z]{2}_', '', col) for col in args.id_cols]
    id_cols = args.id_cols
    target_col = args.target
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)

    X = df.drop(columns=[args.target]+id_cols) #+id_cols_fixed)
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    """
    KNN Regressor tunable params:
    # weights: distance. metric: euclidean, chebyshev, mahalanobis, rogerstanimoto, l2, manhattan, yule, seuclidean,
    # hamming, canberra, correlation, dice, precomputed, sokalmichener, cityblock, sokalsneath, jaccard, l1, p,
    # chebyshev, sqeuclidean, man_euclidean, cosine , braycurtis, russellrao, minkowski, infinity, haversine
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    rf = RandomForestRegressor(random_state=42)
    selector = RFECV(estimator=rf, step=10, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    selector.fit(X_train_processed, y_train)

    # Apply feature mask from RFECV
    X_train_selected = selector.transform(X_train_processed)
    X_test_selected = selector.transform(X_test_processed)

    # Save model framework as numpy arrays for model architecture proofs
    np.save(out_dir / 'X_train_selected.npy', X_train_selected)
    np.save(out_dir / 'X_test_selected.npy', X_test_selected)
    np.save(out_dir / 'y_train.npy', y_train)
    np.save(out_dir / 'y_test.npy', y_test)

    # Train final KNN on selected features
    final_model = KNeighborsRegressor(n_neighbors=6, weights='distance', metric='braycurtis')
    final_model.fit(X_train_selected, y_train)
    y_pred = final_model.predict(X_test_selected)
    np.save(out_dir / 'y_pred.npy', y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_train_selected.shape[1]
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1) if n > p + 1 else None

    # Training metrics
    y_train_pred = final_model.predict(X_train_selected)
    np.save(out_dir / 'y_train_pred.npy', y_train_pred)
    n_tr = len(y_train)
    p_tr = X_train_selected.shape[1]
    train_metrics = {
        'R2': r2_score(y_train, y_train_pred),
        'Adjusted R2': 1 - (1 - r2_score(y_train, y_train_pred)) * (n_tr - 1) / (n_tr - p_tr - 1) if n_tr > p_tr + 1 else None,
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }

    # Test metrics
    y_test_pred = final_model.predict(X_test_selected)
    np.save(out_dir / 'y_test_pred.npy', y_test_pred)
    n_te = len(y_test)
    p_te = X_test_selected.shape[1]
    test_metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'Adjusted R2': 1 - (1 - r2_score(y_test, y_test_pred)) * (n_te - 1) / (n_te - p_te - 1) if n_te > p_te + 1 else None,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }

    # Final model metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Adjusted R2': adj_r2,
        'Selected Features Count': p
    }

    # Build pipeline
    joblib.dump(final_model, out_dir / 'knn_final_model.joblib')
    joblib.dump(preprocessor, out_dir / 'preprocessor.joblib')
    joblib.dump(selector, out_dir / 'rfecv_selector.joblib')

    save_metrics(metrics, out_dir)
    save_metrics_table(train_metrics, test_metrics, out_dir)
    plot_and_save(y_test, y_pred, out_dir)
    plot_true_vs_pred(y_train, y_train_pred, out_dir, 'Training', train_metrics)
    plot_true_vs_pred(y_test, y_test_pred, out_dir, 'Testing', test_metrics)
    plot_feature_importances(selector.estimator_.feature_importances_, feature_names, out_dir)
    plot_statewise_histogram(df, target_col, 'State_Name', out_dir)
    plot_statewise_facets(df, target_col, 'State_Name', out_dir)
    print(json.dumps(metrics, indent=2))

    knn_features = selector.estimator_.feature_importances_
    rf_support = selector.get_support()
    rf_preprocessor = preprocessor.get_feature_names_out()
    rf_features = np.array(rf_preprocessor)[rf_support]

    print("\nKNN Feature Importance:")
    for name, importance in zip(feature_names, knn_features):
        print(f"{name}: {importance:.4f}")

    print("\nRF Feature Importance:\n", rf_features)

    # KNN Model Benchmark Testing and Model Validation
    print("\nPreprocessor features:\n", len(preprocessor.get_feature_names_out())) # remove len() to output all feature names
    #print("\nRF boolean support mask:\n:", selector.get_support())
    #print("\nRF boolean support mask length:\n", len(selector.get_support()))
    #print("\nX_train_processed shape:\n", X_train_processed.shape)
    print("\nknn_features:\n", knn_features)
    print("\nLength of knn_features:\n", len(knn_features))
    print("\nrf_features:\n", rf_features)
    print("\nLength of rf_features:\n", len(rf_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust KNN regression with RFECV feature selection')
    parser.add_argument('--data', required=True, help='Dataset path (csv or parquet)')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--id-cols', nargs='+', default=[], help='ID columns to exclude')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--outdir', default='artifacts', help='Output directory')
    args = parser.parse_args()
    main(args)
