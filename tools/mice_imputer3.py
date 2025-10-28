#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


def get_gpu_xgb_estimator(device: str = 'cuda:0', n_jobs: int = 1) -> XGBRegressor:
    try:
        return XGBRegressor(
            tree_method='hist',
            device=device,
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=42,
            verbosity=0
        )
    except Exception as e:
        print(f"  WARNING: Could not create GPU XGBoost estimator: {e}")
        print(f"  Falling back to CPU-only XGBoost")
        return XGBRegressor(
            tree_method='hist',
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=42,
            verbosity=0
        )


def handle_empty_features(df: pd.DataFrame, drop_empty: bool = False) -> pd.DataFrame:
    empty_cols = df.columns[df.isna().all()].tolist()

    if empty_cols:
        if drop_empty:
            print(f"  Dropping {len(empty_cols)} completely empty columns")
            df = df.drop(columns=empty_cols)
        else:
            print(f"  Pre-filling {len(empty_cols)} empty columns with column-wise medians:")
            for col in empty_cols:
                col_median = df[col].median()
                if np.isnan(col_median):
                    col_median = 0
                print(f"    • {col}: {col_median:.4f}")
                df[col] = col_median

    return df


def mice_fit_transform_gpu(
    X: pd.DataFrame,
    max_iter: int = 10,
    tol: float = 1e-3,
    device: str = 'cuda:0',
    n_jobs: int = -1,
    random_state: int = 42,
    progress_msg: str = ""
) -> pd.DataFrame:
    if X.isna().sum().sum() == 0:
        return X

    X = handle_empty_features(X, drop_empty=False)

    estimator = get_gpu_xgb_estimator(device=device, n_jobs=n_jobs)

    imp = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=0,
        imputation_order='ascending'
    )

    try:
        if progress_msg:
            print(f"  {progress_msg}: Running GPU-accelerated MICE...")

        X_imputed = pd.DataFrame(
            imp.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        return X_imputed

    except Exception as e:
        print(f"  WARNING: GPU MICE failed ({e}), falling back to median imputation")
        return median_impute(X)


def median_impute(X: pd.DataFrame) -> pd.DataFrame:
    X = handle_empty_features(X, drop_empty=False)

    imp = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imp.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    return X_imputed


def impute_categorical(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    for col in cat_cols:
        if col not in df.columns:
            continue

        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)

    return df


def impute_block_gpu(
    block: pd.DataFrame,
    id_cols: List[str],
    targets: List[str],
    args: argparse.Namespace,
    progress_msg: str = ""
) -> pd.DataFrame:
    if block.empty:
        return block

    numeric_cols = block.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = block.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in id_cols]
    cat_cols = [c for c in cat_cols if c not in id_cols]

    if cat_cols and not getattr(args, 'nocatimpute', False):
        block = impute_categorical(block, cat_cols)

    non_targets = [c for c in numeric_cols if c not in targets]

    if non_targets:
        X_pred = block[non_targets].copy()

        if args.impute_targets == 'median':
            X_pred_imp = median_impute(X_pred)
        else:
            device = getattr(args, 'device', 'cuda:0')
            X_pred_imp = mice_fit_transform_gpu(
                X_pred,
                max_iter=args.mice_max_iter,
                tol=args.mice_tol,
                device=device,
                random_state=args.reg_seed,
                progress_msg=f"{progress_msg} (predictors)"
            )

        block[non_targets] = X_pred_imp

    if targets:
        target_cols_present = [t for t in targets if t in numeric_cols]
        if target_cols_present:
            X_targ = block[target_cols_present].copy()

            if args.impute_targets == 'median':
                X_targ_imp = median_impute(X_targ)
            else:
                device = getattr(args, 'device', 'cuda:0')
                X_targ_imp = mice_fit_transform_gpu(
                    X_targ,
                    max_iter=args.mice_max_iter,
                    tol=args.mice_tol,
                    device=device,
                    random_state=args.reg_seed,
                    progress_msg=f"{progress_msg} (targets)"
                )

            block[target_cols_present] = X_targ_imp

    return block


def impute_dataframe(
    df: pd.DataFrame,
    id_cols: List[str],
    targets: List[str],
    group_by_state: bool = True,
    args: Optional[argparse.Namespace] = None,
    progress: bool = True,
    tag: str = ""
) -> pd.DataFrame:
    if args is None:
        args = argparse.Namespace(
            mice_max_iter=10,
            mice_tol=1e-3,
            reg_seed=42,
            impute_targets='regression',
            nocatimpute=False,
            device='cuda:0'
        )

    df = handle_empty_features(df, drop_empty=False)

    state_col = 'State_Name' if 'State_Name' in df.columns else None

    if group_by_state and state_col and state_col in df.columns:
        pieces = []
        states = df[state_col].unique()

        iterator = tqdm(states, desc=f"{tag}:states") if progress else states

        for state in iterator:
            state_df = df[df[state_col] == state].copy()
            state_imputed = impute_block_gpu(
                state_df,
                id_cols,
                targets,
                args,
                progress_msg=f"{tag}:{state}"
            )
            pieces.append(state_imputed)

        return pd.concat(pieces, ignore_index=True)
    else:
        return impute_block_gpu(df, id_cols, targets, args, progress_msg=tag)


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated MICE Imputation")
    parser.add_argument('--data', required=True, help="Input CSV path")
    parser.add_argument('--output', required=True, help="Output CSV path")
    parser.add_argument('--target', required=True, help="Target column")
    parser.add_argument('--id-cols', nargs='+', default=['State_Name', 'State_District_Name'],
                       help="ID columns")
    parser.add_argument('--mice-max-iter', type=int, default=10, help="MICE max iterations")
    parser.add_argument('--mice-tol', type=float, default=1e-3, help="MICE tolerance")
    parser.add_argument('--impute-targets', default='regression', choices=['median', 'regression'],
                       help="Imputation method")
    parser.add_argument('--device', default='cuda:0', help="GPU device (e.g., 'cuda:0')")
    parser.add_argument('--nocatimpute', action='store_true', help="Skip categorical imputation")
    parser.add_argument('--progress', action='store_true', help="Show progress")
    parser.add_argument('--reg-seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("="*80)
    print("GPU-Accelerated MICE Imputation (v3 - Robust Categorical Handling)")
    print("="*80)

    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print("\nRunning GPU-accelerated MICE imputation...")
    df_imputed = impute_dataframe(
        df,
        id_cols=args.id_cols,
        targets=[args.target],
        group_by_state=True,
        args=args,
        progress=args.progress,
        tag='main'
    )

    print(f"\nSaving to {args.output}...")
    df_imputed.to_csv(args.output, index=False)
    print("✓ Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
