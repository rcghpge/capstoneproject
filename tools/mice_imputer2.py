#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MICE Imputer for AHS State and District datasets (robust, no-leakage by default)

Example Usage
From CSV:
python mice_imputer.py csv \
  --state_csv data/Statewise.csv \
  --district_csv data/Districtwise.csv \
  --out_state_csv out/Statewise_mice.csv \
  --out_district_csv out/Districtwise_mice.csv \
  --target YY_Infant_Mortality_Rate_Imr_Total_Person \
  --progress

From SQLite (.db):
python mice_imputer.py sqlite \
  --db_path data/health.db \
  --state_table Statewise \
  --district_table Districtwise \
  --out_state_csv out/Statewise_mice.csv \
  --out_district_csv out/Districtwise_mice.csv \
  --target YY_Infant_Mortality_Rate_Imr_Total_Person \
  --progress


Flag Reference
Core Modes:
  csv                  Run imputation using CSV input files.
  sqlite               Run imputation using SQLite database tables.

Required (per mode):
  --state_csv           Path to State-level input CSV.
  --district_csv        Path to District-level input CSV.
  --out_state_csv       Output CSV path for State-level imputed data.
  --out_district_csv    Output CSV path for District-level imputed data.
  --db_path             SQLite database path (for sqlite mode only).
  --state_table         Table name for State-level data (sqlite mode only).
  --district_table      Table name for District-level data (sqlite mode only).

General Options:
  --target <col>        Target variable(s) for regression/MICE imputation.
                        Repeatable; defaults to YY_Infant_Mortality_Rate_Imr_Total_Person.
  --progress            Enable tqdm progress bars.
  --no_group_by_state   Disable per-state imputation (global imputation across all rows).
  --no_cat_impute       Skip imputation for categorical columns.
  --no_flags            Do not append "_was_missing" indicator columns.

MICE/Model Options:
  --mice_model          Base estimator used for MICE.
                        Choices: bayesridge (default), huber, ridge, lasso, elasticnet.
  --mice_max_iter       Number of MICE iterations (default: 10).
  --mice_tol            Convergence tolerance (default: 1e-3).
  --reg_seed            Random seed for reproducibility.

Target Imputation Strategy:
  --impute_targets      Strategy for filling target columns:
                          regression (default): per-target robust regressor
                          mice: iterative imputation of targets
                          median: median fill
  --allow_leakage       Allow targets to influence predictor imputations (only with --impute_targets mice).
  --target_model        Override the target regression model
                        (same choices as --mice_model).

Huber Parameters (robust regression tuning):
  --huber_epsilon       Threshold for outlier influence (default: 1.35).
  --huber_alpha         Regularization term (default: 0.0001).
  --huber_max_iter      Max iterations for HuberRegressor (default: 1000).
  --huber_tol           Tolerance for convergence (default: 1e-5).

Ridge/Lasso/ElasticNet Parameters:
  --ridge_alpha         Regularization parameter for Ridge (default: 1.0).
  --lasso_alpha         Regularization strength for Lasso (default: 0.001).
  --lasso_max_iter      Max iterations for Lasso (default: 5000).
  --lasso_tol           Tolerance for Lasso (default: 1e-4).
  --enet_alpha          Regularization strength for ElasticNet (default: 0.001).
  --enet_l1_ratio       L1/L2 mixing ratio for ElasticNet (default: 0.5).
  --enet_max_iter       Max iterations for ElasticNet (default: 5000).
  --enet_tol            Tolerance for ElasticNet (default: 1e-4).

Output:
  Writes two CSVs (state and district) with all imputations complete.
  Adds "_was_missing" flags and restores integer-like columns.
  Progress is displayed as dataset → state → 4-step block tracker → per-target.

Design Summary
- Numeric predictors imputed via IterativeImputer (MICE) with robust linear estimators.
- Targets imputed separately (default regression-based, optionally MICE or median).
- No leakage between targets and predictors by default.
- Categoricals imputed via mode (SimpleImputer) unless disabled.
- Integer-like semantics preserved and missingness flags added.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Iterable, List
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge, Lasso, ElasticNet

DEFAULT_ID_COLS = ["State_Name", "State_District_Name", "State_Code", "District_Code"]
DEFAULT_TARGETS = ["YY_Infant_Mortality_Rate_Imr_Total_Person"]
DEFAULT_MICE = dict(max_iter=10, tol=1e-3)
MIN_OBS_NEEDED = 30


def detect_integer_like_columns(df: pd.DataFrame) -> List[str]:
    int_like = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            int_like.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            non_na = s.dropna()
            if not non_na.empty and np.all(np.isclose(non_na, np.round(non_na))):
                int_like.append(c)
    return int_like


def make_missingness_flags(df_num: pd.DataFrame) -> pd.DataFrame:
    if df_num.empty:
        return pd.DataFrame(index=df_num.index)
    return pd.DataFrame(
        {f"{c}_was_missing": df_num[c].isna().astype("int8") for c in df_num.columns},
        index=df_num.index,
    )


def iter_with_progress(iterable: Iterable, desc: str, total: Optional[int], enable: bool):
    if enable:
        return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)
    return iterable


def build_linear_pipe(name: str, args: argparse.Namespace):
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    if name == "bayesridge":
        base = BayesianRidge()
    elif name == "huber":
        base = HuberRegressor(
            epsilon=args.huber_epsilon, alpha=args.huber_alpha,
            fit_intercept=True, max_iter=args.huber_max_iter, tol=args.huber_tol
        )
    elif name == "ridge":
        base = Ridge(alpha=args.ridge_alpha, random_state=args.reg_seed)
    elif name == "lasso":
        base = Lasso(alpha=args.lasso_alpha, max_iter=args.lasso_max_iter, tol=args.lasso_tol,
                     random_state=args.reg_seed)
    elif name == "elasticnet":
        base = ElasticNet(alpha=args.enet_alpha, l1_ratio=args.enet_l1_ratio,
                          max_iter=args.enet_max_iter, tol=args.enet_tol, random_state=args.reg_seed)
    else:
        raise ValueError(f"Unknown linear model: {name}")
    return make_pipeline(scaler, base)


def build_mice_imputer(args: argparse.Namespace) -> IterativeImputer:
    estimator = build_linear_pipe(args.mice_model, args)
    return IterativeImputer(
        estimator=estimator,
        max_iter=args.mice_max_iter,
        tol=args.mice_tol,
        initial_strategy="median",
        sample_posterior=False,
        random_state=args.reg_seed,
        imputation_order="ascending",
        skip_complete=False,
    )


def build_target_regressor(args: argparse.Namespace):
    return build_linear_pipe(args.target_model or args.mice_model, args)


def mice_fit_transform(df_num: pd.DataFrame, args: argparse.Namespace, progress_msg: Optional[str] = None) -> pd.DataFrame:
    if df_num.empty:
        return pd.DataFrame(index=df_num.index)
    use_cols = [c for c in df_num.columns if not df_num[c].isna().all()]
    dropped = [c for c in df_num.columns if c not in use_cols]
    if not use_cols:
        return df_num.copy()
    X = df_num[use_cols].copy()
    if not X.columns.is_unique:
        X = X.loc[:, ~X.columns.duplicated(keep="first")]
    if args.progress and progress_msg:
        tqdm.write(f"[MICE] {progress_msg}: {len(use_cols)} cols, {len(dropped)} all-NaN skipped")
    imp = build_mice_imputer(args)
    X_imp = pd.DataFrame(imp.fit_transform(X), index=X.index, columns=X.columns)
    for c in dropped:
        X_imp[c] = np.nan
    # Reorder to original column order
    return X_imp.reindex(columns=df_num.columns, fill_value=np.nan)


def regression_impute_targets(
    df_num_with_preds: pd.DataFrame, targets: List[str], args: argparse.Namespace,
    progress: bool = False, tag: Optional[str] = None
) -> pd.DataFrame:
    # Defensive copies & de-dup
    X_all = df_num_with_preds.copy()
    if not X_all.columns.is_unique:
        X_all = X_all.loc[:, ~X_all.columns.duplicated(keep="first")]

    targets = [t for t in (targets or []) if t in X_all.columns]
    non_targets = [c for c in X_all.columns if c not in targets]

    preds = X_all[non_targets] if non_targets else pd.DataFrame(index=X_all.index)
    out = pd.DataFrame(index=X_all.index)

    it = iter_with_progress(targets, desc=f"{tag or 'targets'}", total=len(targets), enable=progress)
    for t in it:
        y = X_all[t]
        if preds.empty:
            out[t] = y.fillna(y.median())
            continue

        ok = preds.notna().all(axis=1)            # Series mask (avoids DataFrame alignment)
        y_aligned = y.loc[preds.index]            # .loc preserves duplicate indices
        obs = y_aligned.notna() & ok
        to_pred = y_aligned.isna() & ok

        if obs.sum() >= MIN_OBS_NEEDED and to_pred.any():
            model = build_target_regressor(args)
            model.fit(preds.loc[obs], y_aligned.loc[obs])
            y_pred = model.predict(preds.loc[to_pred])
            y_out = y_aligned.copy()
            y_out.loc[to_pred] = y_pred
            out[t] = y_out.fillna(y_aligned.median())
        else:
            out[t] = y_aligned.fillna(y_aligned.median())

    # Ensure predictors are preserved for caller if needed
    return pd.concat([preds, out], axis=1)


def impute_block(block: pd.DataFrame, id_cols: List[str], targets: List[str], args: argparse.Namespace,
                 progress: bool, tag: str) -> pd.DataFrame:
    id_cols = [c for c in id_cols if c in block.columns]
    numeric_cols = block.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in block.columns if c not in numeric_cols and c not in id_cols]
    int_like_cols = detect_integer_like_columns(block[numeric_cols]) if numeric_cols else []
    flags = make_missingness_flags(block[numeric_cols]) if (args.add_flags and numeric_cols) else None

    stepbar = tqdm(total=4, desc=f"{tag}:steps", leave=False, dynamic_ncols=True) if progress else None

    def step_done():
        if stepbar is not None:
            stepbar.update(1)

    # 1) Numeric imputation
    if numeric_cols:
        non_targets = [c for c in numeric_cols if c not in targets]
        if args.impute_targets == "mice" and args.allow_leakage:
            num_imp = mice_fit_transform(block[numeric_cols], args, progress_msg=f"{tag} (all numeric)")
        else:
            preds_imp = (
                mice_fit_transform(block[non_targets], args, progress_msg=f"{tag} (predictors only)")
                if non_targets else pd.DataFrame(index=block.index)
            )
            if args.impute_targets == "median" or not targets:
                tcols = [t for t in targets if t in block.columns]
                t_df = block[tcols].copy() if tcols else pd.DataFrame(index=block.index)
                for t in t_df.columns:
                    t_df[t] = t_df[t].fillna(t_df[t].median())
                num_imp = pd.concat([preds_imp, t_df], axis=1)
            elif args.impute_targets == "regression":
                merge = pd.concat(
                    [preds_imp, block[[t for t in targets if t in block.columns]]],
                    axis=1
                )
                num_imp = regression_impute_targets(merge, targets, args, progress=progress, tag=f"{tag}:targets")
            elif args.impute_targets == "mice":
                merge = pd.concat(
                    [preds_imp, block[[t for t in targets if t in block.columns]]],
                    axis=1
                )
                imputed = mice_fit_transform(merge, args, progress_msg=f"{tag} (targets only MICE)")
                # Keep predictor values from preds_imp (no leakage)
                for c in preds_imp.columns:
                    imputed[c] = preds_imp[c]
                num_imp = imputed
            else:
                raise ValueError("Unknown --impute_targets option.")
    else:
        num_imp = pd.DataFrame(index=block.index)
    if not num_imp.columns.is_unique:
        num_imp = num_imp.loc[:, ~num_imp.columns.duplicated(keep="first")]
    step_done()

    # 2) Categorical imputation
    if cat_cols:
        if args.impute_categorical:
            simp = SimpleImputer(strategy="most_frequent")
            cat_imp = pd.DataFrame(simp.fit_transform(block[cat_cols]), columns=cat_cols, index=block.index)
        else:
            cat_imp = block[cat_cols].copy()
    else:
        cat_imp = pd.DataFrame(index=block.index)
    step_done()

    # 3) Combine with ids and restore integer-like
    ids = block[id_cols].copy() if id_cols else pd.DataFrame(index=block.index)
    combined = pd.concat([ids, num_imp, cat_imp], axis=1)
    if not combined.columns.is_unique:
        combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    for c in int_like_cols:
        if c in combined.columns:
            combined[c] = pd.to_numeric(np.round(combined[c]), errors="coerce").astype("Int64")
    step_done()

    # 4) Append missingness flags (avoid dup)
    if flags is not None:
        f = flags.loc[:, ~flags.columns.isin(combined.columns)]
        if not f.empty:
            combined = pd.concat([combined, f], axis=1)
    step_done()

    if stepbar is not None:
        stepbar.close()

    # Preserve original order when possible
    original_order = [c for c in block.columns if c in combined.columns] + \
                     [c for c in combined.columns if c not in block.columns]
    if len(set(original_order)) != len(original_order):
        return combined
    return combined.loc[:, original_order]


def impute_dataframe(df: pd.DataFrame, id_cols: List[str], targets: List[str], group_by_state: bool,
                     args: argparse.Namespace, progress: bool, tag: str) -> pd.DataFrame:
    if group_by_state and "State_Name" in df.columns:
        groups = list(df.groupby("State_Name", dropna=False, sort=False))
        pieces = []
        it = iter_with_progress(groups, desc=f"{tag}:states", total=len(groups), enable=progress)
        for state, g in it:
            # keep original index for safe reassembly
            pieces.append(impute_block(g, id_cols, targets, args, progress, tag=f"{tag}:{state}"))
        out_cat = pd.concat(pieces)
        # Reindex safely back to original row order
        if out_cat.index.is_unique and df.index.is_unique:
            out = out_cat.reindex(df.index)
        else:
            # fall back to loc-based alignment
            out = out_cat.loc[df.index]
    else:
        out = impute_block(df, id_cols, targets, args, progress, tag=f"{tag}:all")
    return out


def read_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_from_sqlite(db_path: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def write_csv(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust MICE Imputer for AHS datasets (with tqdm progress bars)")
    sub = p.add_subparsers(dest="mode", required=True)

    csvp = sub.add_parser("csv", help="Read from CSVs")
    csvp.add_argument("--state_csv", required=True)
    csvp.add_argument("--district_csv", required=True)
    csvp.add_argument("--out_state_csv", required=True)
    csvp.add_argument("--out_district_csv", required=True)

    sqlp = sub.add_parser("sqlite", help="Read from SQLite tables")
    sqlp.add_argument("--db_path", required=True)
    sqlp.add_argument("--state_table", required=True)
    sqlp.add_argument("--district_table", required=True)
    sqlp.add_argument("--out_state_csv", required=True)
    sqlp.add_argument("--out_district_csv", required=True)

    p.add_argument("--target", action="append", default=DEFAULT_TARGETS, help="Target column(s). Repeatable.")
    p.add_argument("--no_group_by_state", action="store_true", help="Disable per-state imputation")
    p.add_argument("--no_cat_impute", action="store_true", help="Do not impute categoricals")
    p.add_argument("--no_flags", action="store_true", help="Do not append _was_missing flags")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")

    p.add_argument("--mice_model", choices=["bayesridge", "huber", "ridge", "lasso", "elasticnet"], default="bayesridge")
    p.add_argument("--mice_max_iter", type=int, default=DEFAULT_MICE["max_iter"])
    p.add_argument("--mice_tol", type=float, default=DEFAULT_MICE["tol"])
    p.add_argument("--reg_seed", type=int, default=42)

    p.add_argument("--impute_targets", choices=["regression", "mice", "median"], default="regression")
    p.add_argument("--allow_leakage", action="store_true")
    p.add_argument("--target_model", choices=["bayesridge", "huber", "ridge", "lasso", "elasticnet"], default=None)

    p.add_argument("--huber_epsilon", type=float, default=1.35)
    p.add_argument("--huber_alpha", type=float, default=0.0001)
    p.add_argument("--huber_max_iter", type=int, default=1000)
    p.add_argument("--huber_tol", type=float, default=1e-5)

    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--lasso_alpha", type=float, default=0.001)
    p.add_argument("--lasso_max_iter", type=int, default=5000)
    p.add_argument("--lasso_tol", type=float, default=1e-4)

    p.add_argument("--enet_alpha", type=float, default=0.001)
    p.add_argument("--enet_l1_ratio", type=float, default=0.5)
    p.add_argument("--enet_max_iter", type=int, default=5000)
    p.add_argument("--enet_tol", type=float, default=1e-4)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    args.add_flags = not args.no_flags
    args.impute_categorical = not args.no_cat_impute
    group_by_state = not args.no_group_by_state
    targets = args.target or []

    if args.mode == "csv":
        df_state = read_from_csv(args.state_csv)
        df_dist = read_from_csv(args.district_csv)
    else:
        df_state = read_from_sqlite(args.db_path, args.state_table)
        df_dist = read_from_sqlite(args.db_path, args.district_table)

    datasets = [("state", df_state, args.out_state_csv), ("district", df_dist, args.out_district_csv)]
    ds_iter = tqdm(datasets, desc="Datasets", total=len(datasets), dynamic_ncols=True) if args.progress else datasets

    for name, df_in, out_path in ds_iter:
        if args.progress:
            tqdm.write(f"[INFO] Processing {name.title()} dataset ({len(df_in):,} rows, {df_in.shape[1]} cols)...")
        imputed = impute_dataframe(
            df_in.copy(),
            DEFAULT_ID_COLS,
            targets,
            group_by_state,
            args,
            progress=args.progress,
            tag=name
        )
        write_csv(imputed, out_path)
        (tqdm.write if args.progress else print)(f"[OK] Saved: {out_path}")

    print("[OK] MICE Imputer Complete!")


if __name__ == "__main__":
    main()
