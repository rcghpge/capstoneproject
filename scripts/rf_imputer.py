#!/usr/bin/env python3
"""
Random-Forest (Tree-Based) RF Imputer for AHS State & District datasets.

Design:
- Numeric NON-target predictors: IterativeImputer(estimator=RandomForestRegressor)  -> MissForest-like.
- Targets: per-target RandomForestRegressor trained only on observed y, using the imputed predictors.
- Categoricals: SimpleImputer(most_frequent) by default (can be disabled or redesigned).
- Optional per-state imputation, missingness flags, and integer-like restoration.

Usage:
CSV:
python rf_imputer.py csv \
  --state_csv data/Statewise.csv \
  --district_csv data/Districtwise.csv \
  --out_state_csv out/Statewise_rf.csv \
  --out_district_csv out/Districtwise_rf.csv

SQLite (.db):
python rf_imputer.py sqlite \
  --db_path data/health.db \
  --state_table Statewise \
  --district_table Districtwise \
  --out_state_csv out/Statewise_rf.csv \
  --out_district_csv out/Districtwise_rf.csv

Optional Flag:
Disable per-state groups
... --no_group_by_state

Don’t add _was_missing flags
... --no_flags

Leave categorical feature columns untouched
... --no_cat_impute

Hyperparameters (Optional)
... --rf_estimators 500 --rf_min_leaf 2 --rf_max_features sqrt --rf_seed 1337 \
    --mf_max_iter 20 --mf_tol 1e-4

Notes:
- Trees do not require scaling (unlike KNN). Robust scaling removed.
- Feature columns that are entirely null/NaN cannot be imputed reliably; we can keep them as NaN (and targets fall
back to median or redesign).
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# -------------------- Preset Defaults --------------------
DEFAULT_ID_COLS = ["State_Name", "State_District_Name", "State_Code", "District_Code"]
DEFAULT_TARGETS = ["YY_Infant_Mortality_Rate_Imr_Total_Person"]

# RF defaults (tunable via CLI)
DEFAULT_RF = dict(
    n_estimators=300,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
)

# MissForest-like defaults (tunable via CLI)
DEFAULT_MF = dict(
    max_iter=10,
    tol=1e-3,
)

# Optional Imputation Features
ADD_MISSINGNESS_FLAGS = True
IMPUTE_CATEGORICAL = True
GROUP_BY_STATE = True
MIN_GROUP_ROWS = 15  # adjust per dataset size
# -------------------------------------------------------------


def detect_integer_like_columns(df: pd.DataFrame):
    """Columns that look like integers (pandas integer dtype or numeric with no fractional part)."""
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


def make_missingness_flags(df_num: pd.DataFrame):
    return pd.DataFrame(
        {f"{c}_was_missing": df_num[c].isna().astype("int8") for c in df_num.columns},
        index=df_num.index,
    )


def iter_with_progress(iterable: Iterable, desc: str, total: Optional[int], enable: bool):
    if enable:
        return tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


def missforest_impute_predictors(
    df_num_non_targets: pd.DataFrame,
    rf_params: dict,
    mf_params: dict,
    progress: bool = False,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """
    MissForest-like iterative imputation for numeric non-target predictors utilizing RandomForestRegressor.
    Columns that are entirely null/NaN are left as NaN and re-attached/added back.
    """
    if df_num_non_targets.empty:
        return pd.DataFrame(index=df_num_non_targets.index)

    # Drop columns that are entirely NaN (IterativeImputer can't infer them or redesign)
    use_cols = [c for c in df_num_non_targets.columns if not df_num_non_targets[c].isna().all()]
    dropped_all_nan = [c for c in df_num_non_targets.columns if c not in use_cols]

    if not use_cols:
        return df_num_non_targets.copy()

    X_use = df_num_non_targets[use_cols].copy()

    if progress:
        tqdm.write(f"[{tag or 'missforest'}] start: {len(use_cols)} cols, {len(dropped_all_nan)} all-NaN skipped")

    rf = RandomForestRegressor(**rf_params, n_jobs=-1)

    imp = IterativeImputer(
        estimator=rf,
        max_iter=mf_params.get("max_iter", DEFAULT_MF["max_iter"]),
        tol=mf_params.get("tol", DEFAULT_MF["tol"]),
        initial_strategy="median",
        sample_posterior=False,
        random_state=rf_params.get("random_state", DEFAULT_RF["random_state"]),
    )

    X_imp = pd.DataFrame(
        imp.fit_transform(X_use),
        index=X_use.index,
        columns=X_use.columns,
    )

    if progress:
        tqdm.write(f"[{tag or 'missforest'}] done")

    # Reattach any all-NaN/null columns as NaN
    for c in dropped_all_nan:
        X_imp[c] = np.nan

    # Set columns back in original order
    X_imp = X_imp[df_num_non_targets.columns]
    return X_imp


def rf_impute_numeric_block(
    df_num: pd.DataFrame,
    targets: list[str] | None,
    rf_params: dict,
    progress: bool = False,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """
    Leakage-safe numeric imputation:
      - Step 1: assumes non-target predictors are already imputed (or passes through to MissForest if needed).
      - Step 2: For each target, train RF on rows with observed y utilizing the imputed predictors; predict missing y.
                Fallback to median if insufficient ground truth or incomplete predictors (or re-implement).
    """
    targets = [t for t in (targets or []) if t in df_num.columns]
    non_targets = [c for c in df_num.columns if c not in targets]

    # If predictors are not imputed yet, check imputations it here:
    if non_targets:
        X_imp = missforest_impute_predictors(
            df_num[non_targets], rf_params, DEFAULT_MF, progress=progress, tag=f"{tag or 'rf'}:missforest"
        )
    else:
        X_imp = pd.DataFrame(index=df_num.index)

    out = pd.concat([X_imp], axis=1)

    # Per-target RF fits with a progress bar
    it_targets = iter_with_progress(targets, desc=f"{tag or 'targets'}", total=len(targets), enable=progress)
    for t in it_targets:
        y = df_num[t]
        if X_imp.empty:
            out[t] = y.fillna(y.median())
            continue

        predictors_ok = X_imp.notna().all(axis=1)
        obs_mask = y.notna() & predictors_ok
        pred_mask = y.isna() & predictors_ok

        # Heuristic: needs enough observed samples to learn for a useful/baseline model
        min_obs_needed = 30
        if obs_mask.sum() >= min_obs_needed and pred_mask.any():
            rf_t = RandomForestRegressor(**rf_params, n_jobs=-1)
            rf_t.fit(X_imp.loc[obs_mask], y.loc[obs_mask])

            y_pred = rf_t.predict(X_imp.loc[pred_mask])

            y_imputed = y.copy()
            y_imputed.loc[pred_mask] = y_pred

            # Any remaining nulls/NaNs → median fallback
            if (~predictors_ok & y_imputed.isna()).any():
                y_imputed = y_imputed.fillna(y.median())

            out[t] = y_imputed
        else:
            out[t] = y.fillna(y.median())

    return out


def impute_dataframe(
    df: pd.DataFrame,
    id_cols: list[str],
    targets: list[str],
    group_by_state: bool,
    add_missing_flags: bool,
    impute_categorical: bool,
    rf_params: dict,
    mf_params: dict,
    progress: bool = False,
    tag: Optional[str] = None,
):
    id_cols = [c for c in id_cols if c in df.columns]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols and c not in id_cols]

    # Preserve integer-like columns
    int_like_cols = detect_integer_like_columns(df[numeric_cols]) if numeric_cols else []

    # Missingness flags (numeric only)
    flags = make_missingness_flags(df[numeric_cols]) if (add_missing_flags and numeric_cols) else None

    def _impute_block(block: pd.DataFrame, block_tag: str, progress: bool):
        # Numeric block
        if numeric_cols:
            # MissForest for non-target predictors
            non_targets = [c for c in numeric_cols if c not in targets]
            if non_targets:
                predictors_imp = missforest_impute_predictors(
                    block[non_targets], rf_params, mf_params, progress=progress, tag=f"{block_tag}:missforest"
                )
            else:
                predictors_imp = pd.DataFrame(index=block.index)

            # Merge predictors with targets (targets may still be null/NaN)
            if targets:
                merge_targets = [t for t in targets if t in block.columns]
                merged_num = pd.concat([predictors_imp, block[merge_targets]], axis=1)
            else:
                merged_num = predictors_imp

            num_imputed = rf_impute_numeric_block(
                merged_num, targets, rf_params, progress=progress, tag=f"{block_tag}:targets"
            )
        else:
            num_imputed = pd.DataFrame(index=block.index)

        # Categorical feature checks
        if cat_cols:
            if impute_categorical:
                simp = SimpleImputer(strategy="most_frequent")
                cat_imp = pd.DataFrame(
                    simp.fit_transform(block[cat_cols]),
                    columns=cat_cols,
                    index=block.index
                )
            else:
                cat_imp = block[cat_cols]
        else:
            cat_imp = pd.DataFrame(index=block.index)

        # IDs unchanged
        ids = block[id_cols] if id_cols else pd.DataFrame(index=block.index)

        # Combine
        combined = pd.concat([ids, num_imputed, cat_imp], axis=1)

        # Round back integer-like (nullable ints so NaNs are preserved)
        for c in int_like_cols:
            if c in combined.columns:
                combined[c] = np.round(combined[c]).astype("Int64")

        return combined

    if group_by_state and "State_Name" in df.columns:
        groups = list(df.groupby("State_Name", dropna=False))
        out_pieces = []
        for state, g in iter_with_progress(groups, desc=f"{tag or 'data'}:states", total=len(groups), enable=progress):
            out_pieces.append(_impute_block(g, block_tag=f"{tag or 'data'}:{state}", progress=progress))
        out = pd.concat(out_pieces).loc[df.index]
    else:
        out = _impute_block(df, block_tag=f"{tag or 'data'}:all", progress=progress)

    # Re-attach flags (optional)
    if flags is not None:
        out = pd.concat([out, flags], axis=1)

    # Keep original column order where possible; append new flags at end
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


def main():
    parser = argparse.ArgumentParser(description="Random-Forest (Tree) Impute AHS State & District datasets")
    sub = parser.add_subparsers(dest="mode", required=True)

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

    parser.add_argument("--target", action="append", default=DEFAULT_TARGETS,
                        help="Target column(s) to EXCLUDE from MissForest step. Multiple --target allowed.")
    parser.add_argument("--no_group_by_state", action="store_true", help="Disable per-state imputation")
    parser.add_argument("--no_flags", action="store_true", help="Disable _was_missing flags")
    parser.add_argument("--no_cat_impute", action="store_true", help="Do not impute categoricals")

    # RF / MissForest knobs (optional)
    parser.add_argument("--rf_estimators", type=int, default=DEFAULT_RF["n_estimators"])
    parser.add_argument("--rf_min_leaf", type=int, default=DEFAULT_RF["min_samples_leaf"])
    parser.add_argument("--rf_max_features", type=str, default=str(DEFAULT_RF["max_features"]))
    parser.add_argument("--rf_seed", type=int, default=DEFAULT_RF["random_state"])

    parser.add_argument("--mf_max_iter", type=int, default=DEFAULT_MF["max_iter"])
    parser.add_argument("--mf_tol", type=float, default=DEFAULT_MF["tol"])

    # Progress bars
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars during imputation")

    args = parser.parse_args()

    # Apply overrides locally (no globals)
    rf_params = dict(
        n_estimators=args.rf_estimators,
        min_samples_leaf=args.rf_min_leaf,
        max_features=args.rf_max_features,
        random_state=args.rf_seed,
    )
    mf_params = dict(
        max_iter=args.mf_max_iter,
        tol=args.mf_tol,
    )

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

    # Top-level dataset loop with progress
    datasets = [
        ("state", df_state, args.out_state_csv),
        ("district", df_dist, args.out_district_csv),
    ]

    results = {}
    for name, df_in, out_path in iter_with_progress(datasets, "datasets", total=len(datasets), enable=args.progress):
        imputed = impute_dataframe(
            df_in,
            DEFAULT_ID_COLS,
            targets,
            group_by_state,
            add_flags,
            impute_categorical,
            rf_params,
            mf_params,
            progress=args.progress,
            tag=name,
        )
        write_csv(imputed, out_path)
        results[name] = out_path

    print("[OK] RF Imputer Complete! Wrote:")
    for k, v in results.items():
        print(f"  {v}")


if __name__ == "__main__":
    main()

