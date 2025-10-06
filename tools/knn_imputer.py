#!/usr/bin/env python3
"""
KNN Imputer for ASH State and District datasets

Example Usage
From CSV:
```bash
python impute_knn_state_and_district.py csv \
  --state_csv data/Statewise.csv \
  --district_csv data/Districtwise.csv \
  --out_state_csv out/Statewise_knn.csv \
  --out_district_csv out/Districtwise_knn.csv
```

From SQLite (.db):
```bash
python impute_knn_state_and_district.py sqlite \
  --db_path data/health.db \
  --state_table Statewise \
  --district_table Districtwise \
  --out_state_csv out/Statewise_knn.csv \
  --out_district_csv out/Districtwise_knn.csv
```

Toggle Flags:
Disable per-state groups (faster baseline)
```python-repl
... --no_group_by_state
```

Don't add `_was_missing` flags
```python-repl
... --no_flags
```

Leave categoricals untouched
```python-repl
... --no_cat_impute
```

Add/adjust targets to exclude from distances
```bash
... --target YY_Infant_Mortality_Rate_Imr_Total_Person --target Some_Other_Target
```
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer, SimpleImputer


# -------------------- Default Preset --------------------
DEFAULT_ID_COLS = ["State_Name", "State_District_Name", "State_Code", "District_Code"]
DEFAULT_TARGETS = ["YY_Infant_Mortality_Rate_Imr_Total_Person"]  # add more if needed
N_NEIGHBORS = 5
WEIGHTS = "distance"
ADD_MISSINGNESS_FLAGS = True
IMPUTE_CATEGORICAL = True   # set False to leave categoricals as-is

# If True, impute within each State_Name group (fallback to global if group is too small)
GROUP_BY_STATE = True
MIN_GROUP_ROWS = N_NEIGHBORS + 1
# -------------------------------------------------------------


def robust_scale_with_nans(df_num: pd.DataFrame):
    """Return scaled df, plus medians and IQR for unscale."""
    med = df_num.median(axis=0, skipna=True)
    q75 = df_num.quantile(0.75)
    q25 = df_num.quantile(0.25)
    iqr = (q75 - q25).replace(0, 1.0)  # avoid division by zero
    scaled = (df_num - med) / iqr
    return scaled, med, iqr


def robust_unscale(df_scaled: pd.DataFrame, med: pd.Series, iqr: pd.Series):
    return df_scaled * iqr + med


def detect_integer_like_columns(df: pd.DataFrame):
    """Columns that look like integers (originally all integers or numeric with no fractional part)."""
    int_like = []
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            int_like.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if not s.empty and np.all(np.isclose(s, np.round(s))):
                int_like.append(c)
    return int_like


def make_missingness_flags(df_num: pd.DataFrame):
    flags = pd.DataFrame(
        {f"{c}_was_missing": df_num[c].isna().astype("int8") for c in df_num.columns},
        index=df_num.index,
    )
    return flags


def knn_impute_numeric_block(df_num: pd.DataFrame, targets: list[str] | None):
    """
    Leakage-safe numeric imputation:
      - Step 1: Impute NON-target numeric columns with KNNImputer.
      - Step 2: Impute each target with KNN regression using the imputed non-targets as features.
    """
    targets = [t for t in (targets or []) if t in df_num.columns]
    non_targets = [c for c in df_num.columns if c not in targets]

    # --- Step 1: predictors (non-targets) ---
    X = df_num[non_targets].copy()

    # drop columns that are entirely NaN to avoid sklearn shrinking the matrix silently
    non_all_nan = [c for c in X.columns if not X[c].isna().all()]
    dropped_all_nan = [c for c in X.columns if c not in non_all_nan]
    X_use = X[non_all_nan]

    if not X_use.empty:
        X_scaled, med_X, iqr_X = robust_scale_with_nans(X_use)
        imputer = KNNImputer(n_neighbors=N_NEIGHBORS, weights=WEIGHTS)
        X_imp_scaled = imputer.fit_transform(X_scaled)
        X_imp = robust_unscale(pd.DataFrame(X_imp_scaled, columns=X_use.columns, index=X_use.index), med_X, iqr_X)
    else:
        X_imp = X_use.copy()

    # reattach any all-NaN predictor columns (they remain NaN)
    for c in dropped_all_nan:
        X_imp[c] = np.nan

    # restore original predictor column order
    X_imp = X_imp[X.columns]

    # --- Step 2: targets via KNN regression on imputed predictors ---
    out = pd.concat([X_imp], axis=1)

    for t in targets:
        y = df_num[t]
        # If the target has no observed values, we cannot train a KNN regressor -> leave as-is
        obs_mask = y.notna() & X_imp.notna().all(axis=1)
        pred_mask = y.isna() & X_imp.notna().all(axis=1)

        if obs_mask.sum() >= max(2, N_NEIGHBORS) and pred_mask.any():
            # scale features like Step 1 (important for distance)
            Xtr_scaled, med_tr, iqr_tr = robust_scale_with_nans(X_imp.loc[:, :])
            # Train on observed rows only
            knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS, weights=WEIGHTS)
            knn.fit(Xtr_scaled.loc[obs_mask], y.loc[obs_mask])

            # Predict missing targets
            y_pred = knn.predict(Xtr_scaled.loc[pred_mask])
            y_imputed = y.copy()
            y_imputed.loc[pred_mask] = y_pred
            out[t] = y_imputed
        else:
            # Not enough ground truth to train -> fall back to median
            out[t] = y.fillna(y.median())
    return out


def impute_dataframe(
    df: pd.DataFrame,
    id_cols: list[str],
    targets: list[str],
    group_by_state: bool,
    add_missing_flags: bool,
    impute_categorical: bool
):
    id_cols = [c for c in id_cols if c in df.columns]
    # Split types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols and c not in id_cols]

    # Preserve integer-like columns
    int_like_cols = detect_integer_like_columns(df[numeric_cols]) if numeric_cols else []

    # Missingness flags (numeric only)
    flags = make_missingness_flags(df[numeric_cols]) if (add_missing_flags and numeric_cols) else None

    def _impute_block(block: pd.DataFrame):
        # Numeric block
        if numeric_cols:
            num_imputed = knn_impute_numeric_block(block[numeric_cols], targets)
        else:
            num_imputed = pd.DataFrame(index=block.index)

        # Categoricals
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

        # Round back integer-like
        for c in int_like_cols:
            if c in combined.columns:
                combined[c] = np.round(combined[c]).astype("Int64")  # nullable int

        return combined

    if group_by_state and "State_Name" in df.columns:
        out_pieces = []
        for state, g in df.groupby("State_Name", dropna=False):
            if len(g) >= MIN_GROUP_ROWS:
                out_pieces.append(_impute_block(g))
            else:
                # Fallback to global model for tiny groups
                out_pieces.append(_impute_block(g))
        out = pd.concat(out_pieces).loc[df.index]
    else:
        out = _impute_block(df)

    # Re-attach flags (optional)
    if flags is not None:
        out = pd.concat([out, flags], axis=1)

    # Keep original column order where possible; append new flags at end
    original_order = [c for c in df.columns if c in out.columns] + [c for c in out.columns if c not in df.columns]
    out = out[original_order]

    return out


def read_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_from_sqlite(db_path: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def write_csv(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="KNN Impute State & District datasets")
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
                        help="Target column(s) to exclude from KNN distances. Use multiple --target flags to add more.")
    parser.add_argument("--no_group_by_state", action="store_true", help="Disable per-state imputation")
    parser.add_argument("--no_flags", action="store_true", help="Disable _was_missing flags")
    parser.add_argument("--no_cat_impute", action="store_true", help="Do not impute categoricals")

    args = parser.parse_args()

    targets = args.target or []
    group_by_state = not args.no_group_by_state
    add_flags = not args.no_flags
    impute_categorical = not args.no_cat_impute

    if args.mode == "csv":
        df_state = read_from_csv(args.state_csv)
        df_dist = read_from_csv(args.district_csv)
    else:
        df_state = read_from_sqlite(args.db_path, args.state_table)
        df_dist = read_from_sqlite(args.db_path, args.district_table)

    imputed_state = impute_dataframe(
        df_state, DEFAULT_ID_COLS, targets, group_by_state, add_flags, impute_categorical
    )
    imputed_dist = impute_dataframe(
        df_dist, DEFAULT_ID_COLS, targets, group_by_state, add_flags, impute_categorical
    )

    write_csv(imputed_state, args.out_state_csv)
    write_csv(imputed_dist, args.out_district_csv)

    print(f"[OK] KNN Imputer Complete! Wrote:\n  {args.out_state_csv}\n  {args.out_district_csv}")


if __name__ == "__main__":
    main()


