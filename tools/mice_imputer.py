#!/usr/bin/env python3
"""
MICE Imputer (robust, no-leakage by default) for AHS State & District datasets
with detailed tqdm progress bars.

See header of previous version for full usage. This build adds:
- Dataset-level tqdm bar
- Per-state tqdm bar
- Per-block step tracker tqdm bar (4 steps)
- Per-target tqdm bar
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
    BayesianRidge,
    HuberRegressor,
    Ridge,
    Lasso,
    ElasticNet,
)

# -------------------- Preset Defaults --------------------
DEFAULT_ID_COLS = ["State_Name", "State_District_Name", "State_Code", "District_Code"]
DEFAULT_TARGETS = ["YY_Infant_Mortality_Rate_Imr_Total_Person"]

DEFAULT_MICE = dict(max_iter=10, tol=1e-3)

ADD_MISSINGNESS_FLAGS = True
IMPUTE_CATEGORICAL = True
GROUP_BY_STATE = True

MIN_OBS_NEEDED = 30
# ---------------------------------------------------------


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
    return pd.DataFrame({f"{c}_was_missing": df_num[c].isna().astype("int8") for c in df_num.columns},
                        index=df_num.index)


def iter_with_progress(iterable: Iterable, desc: str, total: Optional[int], enable: bool):
    if enable:
        return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)
    return iterable


# ---------- Estimator builders ----------
def build_linear_pipe(name: str, args: argparse.Namespace):
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    if name == "bayesridge":
        base = BayesianRidge()
    elif name == "huber":
        base = HuberRegressor(
            epsilon=args.huber_epsilon,
            alpha=args.huber_alpha,
            fit_intercept=True,
            max_iter=args.huber_max_iter,
            tol=args.huber_tol,
        )
    elif name == "ridge":
        base = Ridge(alpha=args.ridge_alpha, random_state=args.reg_seed)
    elif name == "lasso":
        base = Lasso(alpha=args.lasso_alpha, max_iter=args.lasso_max_iter, tol=args.lasso_tol,
                     random_state=args.reg_seed)
    elif name == "elasticnet":
        base = ElasticNet(alpha=args.enet_alpha, l1_ratio=args.enet_l1_ratio, max_iter=args.enet_max_iter,
                          tol=args.enet_tol, random_state=args.reg_seed)
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


# ---------- Core steps ----------
def mice_fit_transform(df_num: pd.DataFrame, args: argparse.Namespace, progress_msg: Optional[str] = None) -> pd.DataFrame:
    """Run MICE on df_num (all provided columns), preserving all-NaN columns."""
    if df_num.empty:
        return pd.DataFrame(index=df_num.index)

    use_cols = [c for c in df_num.columns if not df_num[c].isna().all()]
    dropped = [c for c in df_num.columns if c not in use_cols]
    if not use_cols:
        return df_num.copy()

    X = df_num[use_cols].copy()
    if args.progress and progress_msg:
        tqdm.write(f"[MICE] {progress_msg}: {len(use_cols)} cols, {len(dropped)} all-NaN skipped")

    imp = build_mice_imputer(args)
    X_imp = pd.DataFrame(imp.fit_transform(X), index=X.index, columns=X.columns)

    for c in dropped:
        X_imp[c] = np.nan
    return X_imp[df_num.columns]


def regression_impute_targets(
    df_num_with_preds: pd.DataFrame,
    targets: List[str],
    args: argparse.Namespace,
    progress: bool = False,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """Per-target robust regression using imputed predictors; fallback to median."""
    targets = [t for t in (targets or []) if t in df_num_with_preds.columns]
    non_targets = [c for c in df_num_with_preds.columns if c not in targets]

    X = df_num_with_preds[non_targets].copy() if non_targets else pd.DataFrame(index=df_num_with_preds.index)
    out = pd.concat([X], axis=1)

    it = iter_with_progress(targets, desc=f"{tag or 'targets'}", total=len(targets), enable=progress)
    for t in it:
        y = df_num_with_preds[t]
        if X.empty:
            out[t] = y.fillna(y.median())
            continue

        ok = X.notna().all(axis=1)
        obs = y.notna() & ok
        to_pred = y.isna() & ok

        if obs.sum() >= MIN_OBS_NEEDED and to_pred.any():
            model = build_target_regressor(args)
            model.fit(X.loc[obs], y.loc[obs])
            y_pred = model.predict(X.loc[to_pred])

            y_imp = y.copy()
            y_imp.loc[to_pred] = y_pred
            out[t] = y_imp.fillna(y.median())
        else:
            out[t] = y.fillna(y.median())

    return out


def impute_block(
    block: pd.DataFrame,
    id_cols: List[str],
    targets: List[str],
    args: argparse.Namespace,
    progress: bool,
    tag: str,
) -> pd.DataFrame:
    id_cols = [c for c in id_cols if c in block.columns]

    numeric_cols = block.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in block.columns if c not in numeric_cols and c not in id_cols]

    int_like_cols = detect_integer_like_columns(block[numeric_cols]) if numeric_cols else []
    flags = make_missingness_flags(block[numeric_cols]) if (args.add_flags and numeric_cols) else None

    # Step tracker (4 steps)
    stepbar = tqdm(total=4, desc=f"{tag}:steps", leave=False, dynamic_ncols=True) if progress else None
    def step_done():
        if stepbar is not None:
            stepbar.update(1)

    # --- Step 1: MICE on numeric predictors (no-leakage by default)
    if numeric_cols:
        non_targets = [c for c in numeric_cols if c not in targets]

        if args.impute_targets == "mice" and args.allow_leakage:
            num_imp = mice_fit_transform(block[numeric_cols], args, progress_msg=f"{tag} (all numeric)")
        else:
            preds_imp = mice_fit_transform(
                block[non_targets], args, progress_msg=f"{tag} (predictors only)"
            ) if non_targets else pd.DataFrame(index=block.index)

            if args.impute_targets == "median" or not targets:
                tcols = [t for t in targets if t in block.columns]
                t_df = block[tcols].copy() if tcols else pd.DataFrame(index=block.index)
                for t in t_df.columns:
                    t_df[t] = t_df[t].fillna(t_df[t].median())
                num_imp = pd.concat([preds_imp, t_df], axis=1)
            elif args.impute_targets == "regression":
                merge = pd.concat([preds_imp, block[[t for t in targets if t in block.columns]]], axis=1)
                num_imp = regression_impute_targets(merge, targets, args, progress=progress, tag=f"{tag}:targets")
            elif args.impute_targets == "mice":
                merge = pd.concat([preds_imp, block[[t for t in targets if t in block.columns]]], axis=1)
                imputed = mice_fit_transform(merge, args, progress_msg=f"{tag} (targets only MICE)")
                # Preserve predictor values from preds_imp to avoid leakage
                imputed[preds_imp.columns] = preds_imp[preds_imp.columns]
                num_imp = imputed
            else:
                raise ValueError("Unknown --impute_targets option.")
    else:
        num_imp = pd.DataFrame(index=block.index)
    step_done()

    # --- Step 2: Categoricals
    if cat_cols:
        if args.impute_categorical:
            simp = SimpleImputer(strategy="most_frequent")
            cat_imp = pd.DataFrame(simp.fit_transform(block[cat_cols]), columns=cat_cols, index=block.index)
        else:
            cat_imp = block[cat_cols]
    else:
        cat_imp = pd.DataFrame(index=block.index)
    step_done()

    # IDs unchanged
    ids = block[id_cols] if id_cols else pd.DataFrame(index=block.index)

    # Combine
    combined = pd.concat([ids, num_imp, cat_imp], axis=1)

    # --- Step 3: Restore integer-like semantics
    for c in int_like_cols:
        if c in combined.columns:
            combined[c] = np.round(combined[c]).astype("Int64")
    step_done()

    # --- Step 4: Attach flags
    if flags is not None:
        combined = pd.concat([combined, flags], axis=1)
    step_done()

    if stepbar is not None:
        stepbar.close()

    # Preserve original order then new cols (flags)
    original_order = [c for c in block.columns if c in combined.columns] + [c for c in combined.columns if c not in block.columns]
    return combined[original_order]


def impute_dataframe(
    df: pd.DataFrame,
    id_cols: List[str],
    targets: List[str],
    group_by_state: bool,
    args: argparse.Namespace,
    progress: bool,
    tag: str,
) -> pd.DataFrame:
    if group_by_state and "State_Name" in df.columns:
        groups = list(df.groupby("State_Name", dropna=False))
        pieces = []
        it = iter_with_progress(groups, desc=f"{tag}:states", total=len(groups), enable=progress)
        for state, g in it:
            pieces.append(impute_block(g, id_cols, targets, args, progress, tag=f"{tag}:{state}"))
        out = pd.concat(pieces).loc[df.index]
    else:
        out = impute_block(df, id_cols, targets, args, progress, tag=f"{tag}:all")
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
    p = argparse.ArgumentParser(description="Robust MICE Imputer for AHS datasets (with tqdm progress bars)")
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
    p.add_argument("--target", action="append", default=DEFAULT_TARGETS, help="Target column(s). Repeatable.")
    p.add_argument("--no_group_by_state", action="store_true", help="Disable per-state imputation")
    p.add_argument("--no_cat_impute", action="store_true", help="Do not impute categoricals")
    p.add_argument("--no_flags", action="store_true", help="Do not append _was_missing flags")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")

    # MICE base model
    p.add_argument("--mice_model", choices=["bayesridge", "huber", "ridge", "lasso", "elasticnet"], default="bayesridge")
    p.add_argument("--mice_max_iter", type=int, default=DEFAULT_MICE["max_iter"])
    p.add_argument("--mice_tol", type=float, default=DEFAULT_MICE["tol"])
    p.add_argument("--reg_seed", type=int, default=42)

    # Target strategy
    p.add_argument("--impute_targets", choices=["regression", "mice", "median"], default="regression",
                   help="How to fill targets; default avoids leakage.")
    p.add_argument("--allow_leakage", action="store_true",
                   help="Only applies with --impute_targets mice. Allows targets to help impute predictors.")
    p.add_argument("--target_model", choices=["bayesridge", "huber", "ridge", "lasso", "elasticnet"], default=None)

    # Huber params
    p.add_argument("--huber_epsilon", type=float, default=1.35)
    p.add_argument("--huber_alpha", type=float, default=0.0001)
    p.add_argument("--huber_max_iter", type=int, default=1000)
    p.add_argument("--huber_tol", type=float, default=1e-5)

    # Ridge/Lasso/EN params
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

    args.add_flags = (not args.no_flags)
    args.impute_categorical = (not args.no_cat_impute)
    group_by_state = (not args.no_group_by_state)
    targets = args.target or []

    # Read
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

    print(f"[INFO] Robust MICE base={args.mice_model} | targets={args.impute_targets}"
          f"{' (LEAKAGE OK)' if (args.impute_targets=='mice' and args.allow_leakage) else ''}")
    print(f"[INFO] Group by state: {group_by_state} | Flags: {args.add_flags} | "
          f"Impute categoricals: {args.impute_categorical}")
    print(f"[INFO] MICE: max_iter={args.mice_max_iter}, tol={args.mice_tol}, seed={args.reg_seed} | Targets: {targets}")

    ds_iter = tqdm(datasets, desc="Datasets", total=len(datasets), dynamic_ncols=True) if args.progress else datasets
    results: Dict[str, str] = {}

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
            tag=name,
        )

        write_csv(imputed, out_path)
        (tqdm.write if args.progress else print)(f"[OK] Saved: {out_path}")
        results[name] = out_path

    print("[OK] MICE Imputer Complete! Wrote:")
    for k, v in results.items():
        print(f"  {v}")


if __name__ == "__main__":
    main()
