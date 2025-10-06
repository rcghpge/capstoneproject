#!/usr/bin/env python3
"""
Regression Imputer for AHS State & District datasets

Design:
- Numeric NON-target predictors:
    IterativeImputer(estimator=Pipeline(RobustScaler -> RegressionModel))
    -> linear/MICE-like.
- Targets:
    Per-target Pipeline(RobustScaler -> RegressionModel) trained only on observed y,
    using the imputed predictors; predict missing y (fallback to median).
- Categoricals:
    SimpleImputer(most_frequent) by default (can be disabled).
- Optional per-state imputation, missingness flags, and integer-like restoration.

Usage:
CSV:
python regression_imputer.py csv \
  --state_csv data/Statewise.csv \
  --district_csv data/Districtwise.csv \
  --out_state_csv out/Statewise_reg.csv \
  --out_district_csv out/Districtwise_reg.csv

SQLite (.db):
python regression_imputer.py sqlite \
  --db_path data/health.db \
  --state_table Statewise \
  --district_table Districtwise \
  --out_state_csv out/Statewise_reg.csv \
  --out_district_csv out/Districtwise_reg.csv

Optional flags:
  --no_group_by_state     # disable per-state groups
  --no_flags              # don't add _was_missing flags
  --no_cat_impute         # leave categoricals untouched

Model selection & imputation features (examples):
  --reg_model huber --huber_epsilon 1.35 --huber_alpha 0.0001 --reg_seed 1337
  --reg_model ridge --ridge_alpha 1.0
  --reg_model lasso --lasso_alpha 0.001 --lasso_max_iter 5000
  --reg_model elasticnet --enet_alpha 0.001 --enet_l1_ratio 0.5 --enet_max_iter 5000
  --reg_model bayesridge
  --mice_max_iter 20 --mice_tol 1e-4

Example Runs:
Robust Huber (default) grouped by state with flags:
python regression_imputer.py \
  --reg_model huber --huber_epsilon 1.35 --huber_alpha 0.0001 --progress \
  csv \
  --state_csv data/Statewise.csv \
  --district_csv data/Districtwise.csv \
  --out_state_csv out/Statewise_reg.csv \
  --out_district_csv out/Districtwise_reg.csv

Ridge (no state grouping, no flags, keep categoricals as-is):
python regression_imputer.py \
  --reg_model ridge --ridge_alpha 1.0 \
  --no_group_by_state --no_flags --no_cat_impute --progress \
  sqlite \
  --db_path data/health.db \
  --state_table Statewise \
  --district_table Districtwise \
  --out_state_csv out/Statewise_reg.csv \
  --out_district_csv out/Districtwise_reg.csv

python tools/regression_imputer.py \
  --reg_model ridge --ridge_alpha 1.0 \
  --no_group_by_state --no_flags --no_cat_impute \
  --progress \
  csv \
  --state_csv data/Key_indicators_statewise.csv \
  --district_csv data/Key_indicators_districtwise.csv \
  --out_state_csv out/Statewise_ridge.csv \
  --out_district_csv out/Districtwise_ridge.csv

Notes:
- Linear models *benefit* from scaling; we use RobustScaler (center=median, scale=IQR).
- Entirely-null columns are left as NaN and reattached.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Iterable, List, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import (
    HuberRegressor,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
)

# -------------------- Preset Defaults --------------------
DEFAULT_ID_COLS = ["State_Name", "State_District_Name", "State_Code", "District_Code"]
DEFAULT_TARGETS = ["YY_Infant_Mortality_Rate_Imr_Total_Person"]

# MICE-like defaults
DEFAULT_MICE = dict(max_iter=10, tol=1e-3)

# Optional features
ADD_MISSINGNESS_FLAGS = True
IMPUTE_CATEGORICAL = True
GROUP_BY_STATE = True

# Heuristic minimum observed rows needed to fit a per-target model
MIN_OBS_NEEDED = 30
# ---------------------------------------------------------


def detect_integer_like_columns(df: pd.DataFrame) -> List[str]:
    """Columns that look like integers (integer dtype or numeric with no fractional part)."""
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
    return pd.DataFrame({f"{c}_was_missing": df_num[c].isna().astype("int8") for c in df_num.columns},
                        index=df_num.index)


def iter_with_progress(iterable: Iterable, desc: str, total: Optional[int], enable: bool):
    if enable:
        # dynamic_ncols keeps stdout bars in varying terminals; leave=False avoids scroll spam in nested loops
        return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)
    return iterable


# ---------- Regression Methods ----------
def build_regressor(args: argparse.Namespace):
    """Create the chosen regression model for targets."""
    if args.reg_model == "huber":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
            HuberRegressor(
                epsilon=args.huber_epsilon,
                alpha=args.huber_alpha,
                fit_intercept=True,
                max_iter=args.huber_max_iter,
                tol=args.huber_tol,
            ),
        )
    elif args.reg_model == "ridge":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
            Ridge(alpha=args.ridge_alpha, random_state=args.reg_seed),
        )
    elif args.reg_model == "lasso":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
            Lasso(alpha=args.lasso_alpha, max_iter=args.lasso_max_iter, tol=args.lasso_tol, random_state=args.reg_seed),
        )
    elif args.reg_model == "elasticnet":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
            ElasticNet(alpha=args.enet_alpha, l1_ratio=args.enet_l1_ratio,
                       max_iter=args.enet_max_iter, tol=args.enet_tol, random_state=args.reg_seed),
        )
    elif args.reg_model == "bayesridge":
        return make_pipeline(
            RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
            BayesianRidge(),
        )
    else:
        raise ValueError(f"Unknown reg_model: {args.reg_model}")


def build_mice_estimator(args: argparse.Namespace):
    """
    Estimator for IterativeImputer on predictors.
    Mirror the target regression method for consistency (wrapped in RobustScaler).
    """
    return build_regressor(args)


# ---------- Imputation steps ----------
def mice_impute_predictors(
    df_num_non_targets: pd.DataFrame,
    args: argparse.Namespace,
    progress: bool = False,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """
    Iterative imputation (MICE-like) for numeric non-target predictors utilizing a robust linear model.
    Entirely-null columns are left as NaN and reattached.
    """
    if df_num_non_targets.empty:
        return pd.DataFrame(index=df_num_non_targets.index)

    use_cols = [c for c in df_num_non_targets.columns if not df_num_non_targets[c].isna().all()]
    dropped_all_nan = [c for c in df_num_non_targets.columns if c not in use_cols]

    if not use_cols:
        return df_num_non_targets.copy()

    X_use = df_num_non_targets[use_cols].copy()

    if progress:
        tqdm.write(f"[{tag or 'mice'}] start: {len(use_cols)} cols, {len(dropped_all_nan)} all-NaN skipped")

    estimator = build_mice_estimator(args)

    imp = IterativeImputer(
        estimator=estimator,
        max_iter=args.mice_max_iter,
        tol=args.mice_tol,
        initial_strategy="median",
        sample_posterior=False,
        random_state=args.reg_seed,
    )

    X_imp = pd.DataFrame(imp.fit_transform(X_use), index=X_use.index, columns=X_use.columns)

    if progress:
        tqdm.write(f"[{tag or 'mice'}] done")

    for c in dropped_all_nan:
        X_imp[c] = np.nan

    X_imp = X_imp[df_num_non_targets.columns]
    return X_imp


def regression_impute_numeric_block(
    df_num: pd.DataFrame,
    targets: List[str],
    args: argparse.Namespace,
    progress: bool = False,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """
    Leakage mitigation for numeric imputation using robust linear models:
      - Assumes non-target predictors have been imputed with MICE already.
      - For each target, fit robust regression on observed y & predictors; predict missing y.
      - Fallback to median if insufficient observations or incomplete predictors.
    """
    targets = [t for t in (targets or []) if t in df_num.columns]
    non_targets = [c for c in df_num.columns if c not in targets]

    X_imp = df_num[non_targets].copy() if non_targets else pd.DataFrame(index=df_num.index)

    out = pd.concat([X_imp], axis=1)

    it_targets = iter_with_progress(targets, desc=f"{tag or 'targets'}", total=len(targets), enable=progress)
    for t in it_targets:
        y = df_num[t]
        if X_imp.empty:
            out[t] = y.fillna(y.median())
            continue

        predictors_ok = X_imp.notna().all(axis=1)
        obs_mask = y.notna() & predictors_ok
        pred_mask = y.isna() & predictors_ok

        if obs_mask.sum() >= MIN_OBS_NEEDED and pred_mask.any():
            model = build_regressor(args)
            model.fit(X_imp.loc[obs_mask], y.loc[obs_mask])
            y_pred = model.predict(X_imp.loc[pred_mask])

            y_imputed = y.copy()
            y_imputed.loc[pred_mask] = y_pred

            # If any missing remain (e.g., predictors not OK), median fallback
            if (~predictors_ok & y_imputed.isna()).any():
                y_imputed = y_imputed.fillna(y.median())

            out[t] = y_imputed
        else:
            out[t] = y.fillna(y.median())

    return out


def impute_dataframe(
    df: pd.DataFrame,
    id_cols: List[str],
    targets: List[str],
    group_by_state: bool,
    add_missing_flags: bool,
    impute_categorical: bool,
    args: argparse.Namespace,
    progress: bool = False,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    id_cols = [c for c in id_cols if c in df.columns]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols and c not in id_cols]

    int_like_cols = detect_integer_like_columns(df[numeric_cols]) if numeric_cols else []

    flags = make_missingness_flags(df[numeric_cols]) if (add_missing_flags and numeric_cols) else None

    def _impute_block(block: pd.DataFrame, block_tag: str, show_progress: bool) -> pd.DataFrame:
        # 1) MICE on non-target predictors
        if numeric_cols:
            non_targets = [c for c in numeric_cols if c not in targets]
            if non_targets:
                predictors_imp = mice_impute_predictors(
                    block[non_targets], args, progress=show_progress, tag=f"{block_tag}:mice"
                )
            else:
                predictors_imp = pd.DataFrame(index=block.index)

            # 2) Merge predictors with (possibly-missing) targets and impute targets via regression
            if targets:
                merge_targets = [t for t in targets if t in block.columns]
                merged_num = pd.concat([predictors_imp, block[merge_targets]], axis=1)
            else:
                merged_num = predictors_imp

            num_imputed = regression_impute_numeric_block(
                merged_num, targets, args, progress=show_progress, tag=f"{block_tag}:targets"
            )
        else:
            num_imputed = pd.DataFrame(index=block.index)

        # 3) Categorical features
        if cat_cols:
            if impute_categorical:
                simp = SimpleImputer(strategy="most_frequent")
                cat_imp = pd.DataFrame(simp.fit_transform(block[cat_cols]),
                                       columns=cat_cols, index=block.index)
            else:
                cat_imp = block[cat_cols]
        else:
            cat_imp = pd.DataFrame(index=block.index)

        # IDs unchanged
        ids = block[id_cols] if id_cols else pd.DataFrame(index=block.index)

        # Combine
        combined = pd.concat([ids, num_imputed, cat_imp], axis=1)

        # 4) Restore integer-like (nullable Int64 to preserve NaN)
        for c in int_like_cols:
            if c in combined.columns:
                combined[c] = np.round(combined[c]).astype("Int64")

        return combined

    if group_by_state and "State_Name" in df.columns:
        groups = list(df.groupby("State_Name", dropna=False))
        out_pieces = []
        for state, g in iter_with_progress(groups, desc=f"{tag or 'data'}:states", total=len(groups), enable=progress):
            out_pieces.append(_impute_block(g, block_tag=f"{tag or 'data'}:{state}", show_progress=progress))
        out = pd.concat(out_pieces).loc[df.index]
    else:
        out = _impute_block(df, block_tag=f"{tag or 'data'}:all", show_progress=progress)

    # Re-attach flags
    if flags is not None:
        out = pd.concat([out, flags], axis=1)

    # Preserve original order, then append new cols (flags)
    original_order = [c for c in df.columns if c in out.columns] + [c for c in out.columns if c not in df.columns]
    out = out[original_order]
    return out


# -------------------- I/O helpers & CLI --------------------
def read_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_from_sqlite(db_path: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def write_csv(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regression Imputer (robust, MICE-style) for AHS datasets")
    sub = p.add_subparsers(dest="mode", required=True)

    # CSV mode
    csvp = sub.add_parser("csv", help="Read from CSVs")
    csvp.add_argument("--state_csv", required=True)
    csvp.add_argument("--district_csv", required=True)
    csvp.add_argument("--out_state_csv", required=True)
    csvp.add_argument("--out_district_csv", required=True)

    # SQLite mode
    sqlp = sub.add_parser("sqlite", help="Read from SQLite tables")
    sqlp.add_argument("--db_path", required=True)
    sqlp.add_argument("--state_table", required=True)
    sqlp.add_argument("--district_table", required=True)
    sqlp.add_argument("--out_state_csv", required=True)
    sqlp.add_argument("--out_district_csv", required=True)

    # Shared options
    p.add_argument("--target", action="append", default=DEFAULT_TARGETS,
                   help="Target column(s) to exclude from MICE step. Multiple --target allowed.")
    p.add_argument("--no_group_by_state", action="store_true", help="Disable per-state imputation")
    p.add_argument("--no_flags", action="store_true", help="Disable _was_missing flags")
    p.add_argument("--no_cat_impute", action="store_true", help="Do not impute categoricals")

    # Model selection
    p.add_argument("--reg_model", choices=["huber", "ridge", "lasso", "elasticnet", "bayesridge"], default="huber")
    p.add_argument("--reg_seed", type=int, default=42)

    # Huber
    p.add_argument("--huber_epsilon", type=float, default=1.35)
    p.add_argument("--huber_alpha", type=float, default=0.0001)
    p.add_argument("--huber_max_iter", type=int, default=1000)
    p.add_argument("--huber_tol", type=float, default=1e-5)

    # Ridge
    p.add_argument("--ridge_alpha", type=float, default=1.0)

    # Lasso
    p.add_argument("--lasso_alpha", type=float, default=0.001)
    p.add_argument("--lasso_max_iter", type=int, default=5000)
    p.add_argument("--lasso_tol", type=float, default=1e-4)

    # ElasticNet
    p.add_argument("--enet_alpha", type=float, default=0.001)
    p.add_argument("--enet_l1_ratio", type=float, default=0.5)
    p.add_argument("--enet_max_iter", type=int, default=5000)
    p.add_argument("--enet_tol", type=float, default=1e-4)

    # MICE
    p.add_argument("--mice_max_iter", type=int, default=DEFAULT_MICE["max_iter"])
    p.add_argument("--mice_tol", type=float, default=DEFAULT_MICE["tol"])

    # Progress
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    targets = args.target or []
    group_by_state = not args.no_group_by_state
    add_flags = not args.no_flags
    impute_categorical = not args.no_cat_impute

    # Read inputs
    if args.mode == "csv":
        df_state = read_from_csv(args.state_csv)
        df_dist = read_from_csv(args.district_csv)
    else:
        df_state = read_from_sqlite(args.db_path, args.state_table)
        df_dist = read_from_sqlite(args.db_path, args.district_table)

    datasets = [
        ("state", df_state, args.out_state_csv),
        ("district", df_dist, args.out_district_csv),
    ]

    # ---------- Top-level info & dataset imputation progress ----------
    model_name = args.reg_model.capitalize()
    print(f"[INFO] Using {model_name} Regression Imputer")
    if args.reg_model == "ridge":
        print(f"[INFO] ridge_alpha={args.ridge_alpha}")
    elif args.reg_model == "huber":
        print(f"[INFO] huber_epsilon={args.huber_epsilon}, huber_alpha={args.huber_alpha}")
    elif args.reg_model == "lasso":
        print(f"[INFO] lasso_alpha={args.lasso_alpha}, max_iter={args.lasso_max_iter}")
    elif args.reg_model == "elasticnet":
        print(f"[INFO] enet_alpha={args.enet_alpha}, l1_ratio={args.enet_l1_ratio}, max_iter={args.enet_max_iter}")

    print(f"[INFO] Mode: {args.mode.upper()} | Group by state: {group_by_state} | Flags: {add_flags} | "
          f"Impute categoricals: {impute_categorical}")
    print(f"[INFO] MICE: max_iter={args.mice_max_iter}, tol={args.mice_tol} | Targets: {targets}")
    print("[INFO] Starting dataset processing...")

    ds_iter = tqdm(datasets, desc="Datasets", total=len(datasets), dynamic_ncols=True) if args.progress else datasets

    results: Dict[str, str] = {}
    for name, df_in, out_path in ds_iter:
        tqdm.write(f"[INFO] Processing {name.title()} dataset ({len(df_in):,} rows, {df_in.shape[1]} cols)...") if args.progress else None

        # 1) Prepare numeric predictors via MICE on non-targets
        numeric_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
        non_targets = [c for c in numeric_cols if c not in targets]
        if non_targets:
            predictors_imp = mice_impute_predictors(
                df_in[non_targets], args=args, progress=args.progress, tag=f"{name}:mice"
            )
            merged_num = predictors_imp
        else:
            merged_num = pd.DataFrame(index=df_in.index)

        # 2) Build a working frame with numeric (predictors + targets) + categorical features + IDs
        work = df_in.copy()
        if not merged_num.empty:
            work[non_targets] = merged_num[non_targets]

        imputed = impute_dataframe(
            work,
            DEFAULT_ID_COLS,
            targets,
            group_by_state,
            add_flags,
            impute_categorical,
            args=args,
            progress=args.progress,
            tag=name,
        )

        write_csv(imputed, out_path)
        tqdm.write(f"[OK] Saved: {out_path}") if args.progress else print(f"[OK] Saved: {out_path}")
        results[name] = out_path

    print("[OK] Regression Imputer Complete! Wrote:")
    for k, v in results.items():
        print(f"  {v}")


if __name__ == "__main__":
    main()
