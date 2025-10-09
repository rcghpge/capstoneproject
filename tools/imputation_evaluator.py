#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified diagnostics for imputation outputs (KNN, MissForest/Tree, Regression, MICE).
- Works with your ORIGINAL CSV (with NaNs) + IMPUTED CSV (same shape/columns).
- Writes plots & metrics to out/diagnostics/<method>/.

Usage examples:
# KNN example
python scripts/impute_diagnostics.py \
  --method knn \
  --original data/Key_indicator_statewise.csv \
  --imputed  out/Statewise_knn.csv \
  --id-cols State_Name State_District_Name \
  --outdir  out/evaluation

# Random Forest (RF) - Tree/MissForest example
python scripts/impute_diagnostics.py \
  --method missforest \
  --original data/Key_indicator_districtwise.csv \
  --imputed  out/Districtwise_rf.csv \
  --id-cols State_Name State_District_Name \
  --outdir  out/evaluation

# Regression imputer example
python scripts/impute_diagnostics.py \
  --method regression \
  --original data/Key_indicator_statewise.csv \
  --imputed  out/Statewise_reg.csv \
  --id-cols State_Name State_District_Name \
  --outdir  out/evaluation

# MICE (IterativeImputer) example
python scripts/impute_diagnostics.py \
  --method mice \
  --original data/Key_indicator_districtwise.csv \
  --imputed  out/Districtwise_mice.csv \
  --id-cols State_Name State_District_Name \
  --outdir  out/evaluation \
  --miceforest-log-dir out/miceforest_kernel   # only if you used miceforest

Notes:
- Iteration traces are only produced if `--miceforest-log-dir` contains a saved
  `miceforest.ImputationKernel` with save_all_iterations=True (optional).
"""

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import missingno as msno
except Exception:
    msno = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _safe_cols(df: pd.DataFrame, id_cols):
    cols = [c for c in df.columns if c not in id_cols]
    return cols

def _subset_numeric(df: pd.DataFrame, cols):
    num_cols = [c for c in cols if _is_numeric(df[c])]
    return num_cols

def _subset_nonid(df: pd.DataFrame, id_cols):
    return df[_safe_cols(df, id_cols)].copy()

def _match_columns(a: pd.DataFrame, b: pd.DataFrame):
    common = [c for c in a.columns if c in b.columns]
    return a[common].copy(), b[common].copy()

def _savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_missingness(original, imputed, outdir):
    if msno is None:
        warnings.warn("missingno not installed; skipping missingness heatmaps.")
        return
    _ensure_dir(outdir)

    plt.figure(figsize=(10,4))
    msno.matrix(original, sparkline=False)
    _savefig(os.path.join(outdir, "missingness_original_matrix.png"))

    plt.figure(figsize=(10,4))
    msno.matrix(imputed, sparkline=False)
    _savefig(os.path.join(outdir, "missingness_imputed_matrix.png"))

    plt.figure(figsize=(8,6))
    msno.heatmap(original)
    _savefig(os.path.join(outdir, "missingness_original_heatmap.png"))

def plot_density_observed_vs_imputed(original, imputed, cols, outdir, max_plots=24):
    _ensure_dir(outdir)
    used = 0
    for col in tqdm(cols, desc="Density plots"):
        if used >= max_plots:
            break
        if not _is_numeric(imputed[col]):
            continue
        obs = original[col].dropna()
        imp = imputed.loc[original[col].isna(), col].dropna()
        if len(obs) == 0 or len(imp) == 0:
            continue

        plt.figure(figsize=(6,4))
        if sns is not None:
            sns.kdeplot(obs, label="Observed", fill=True)
            sns.kdeplot(imp, label="Imputed", fill=True, alpha=0.6)
        else:
            plt.hist(obs, bins=30, alpha=0.6, label="Observed", density=True)
            plt.hist(imp, bins=30, alpha=0.6, label="Imputed", density=True)
        plt.title(f"Observed vs Imputed: {col}")
        plt.legend()
        _savefig(os.path.join(outdir, f"density_{col}.png"))
        used += 1

def plot_qq_and_ks(original, imputed, cols, outdir, max_plots=24):
    _ensure_dir(outdir)
    results = []
    used = 0
    for col in tqdm(cols, desc="Q-Q & KS"):
        if used >= max_plots:
            break
        if not _is_numeric(imputed[col]):
            continue
        obs = original[col].dropna()
        imp = imputed.loc[original[col].isna(), col].dropna()
        if len(obs) < 5 or len(imp) < 5:
            continue

        # Q-Q plot
        plt.figure(figsize=(5,5))
        percs = np.linspace(0.01, 0.99, 100)
        qo = np.quantile(obs, percs)
        qi = np.quantile(imp, percs)
        plt.scatter(qo, qi, s=8)
        lims = [min(qo.min(), qi.min()), max(qo.max(), qi.max())]
        plt.plot(lims, lims, linestyle="--")
        plt.title(f"Q-Q: {col} (Observed vs Imputed)")
        plt.xlabel("Observed quantiles")
        plt.ylabel("Imputed quantiles")
        _savefig(os.path.join(outdir, f"qq_{col}.png"))

        # KS test
        ks_stat, ks_p = stats.ks_2samp(obs, imp, method="asymp")
        results.append({"column": col, "ks_stat": ks_stat, "ks_pvalue": ks_p})
        used += 1

    if results:
        pd.DataFrame(results).to_csv(os.path.join(outdir, "qq_ks_summary.csv"), index=False)

def plot_relationships(original, imputed, cols, outdir, max_pairs=12):
    """Scatter relationships with 'imputed' hue; picks random pairs of numeric columns."""
    _ensure_dir(outdir)
    num_cols = [c for c in cols if _is_numeric(imputed[c])]
    if len(num_cols) < 2:
        return
    rng = np.random.default_rng(123)
    pairs = []
    for _ in range(max_pairs):
        a, b = rng.choice(num_cols, size=2, replace=False)
        if a == b:
            continue
        pairs.append((a,b))
    if not pairs:
        return

    mask_imputed_any = original[num_cols].isna().any(axis=1)
    df_plot = imputed[num_cols].copy()
    df_plot["__imputed_row__"] = np.where(mask_imputed_any, "had_missing", "fully_observed")

    for a,b in tqdm(pairs, desc="Relationships"):
        plt.figure(figsize=(5,5))
        if sns is not None:
            sns.scatterplot(data=df_plot, x=a, y=b, hue="__imputed_row__", s=10, alpha=0.7)
        else:
            for tag, sub in df_plot.groupby("__imputed_row__"):
                plt.scatter(sub[a], sub[b], s=8, alpha=0.6, label=tag)
            plt.legend()
        plt.title(f"Relationship: {a} vs {b}")
        _savefig(os.path.join(outdir, f"pair_{a}__{b}.png"))

def plot_corr_shift(original, imputed, cols, outdir):
    """Compare correlation matrices computed on overlapping nonmissing rows for each pair."""
    _ensure_dir(outdir)
    num_cols = [c for c in cols if _is_numeric(imputed[c])]
    if len(num_cols) < 2:
        return

    complete_mask = ~original[num_cols].isna().any(axis=1)
    orig_clean = original.loc[complete_mask, num_cols]
    imp_clean = imputed[num_cols]

    if len(orig_clean) < 3:
        return

    corr_orig = orig_clean.corr(numeric_only=True)
    corr_imp  = imp_clean.corr(numeric_only=True)
    shift = (corr_imp - corr_orig)

    for matrix, name in [(corr_orig, "corr_original"),
                         (corr_imp,  "corr_imputed"),
                         (shift,     "corr_shift")]:
        plt.figure(figsize=(8,6))
        if sns is not None:
            sns.heatmap(matrix, vmin=-1, vmax=1, center=0, square=True, cbar=True)
        else:
            plt.imshow(matrix, vmin=-1, vmax=1)
            plt.colorbar()
        plt.title(name)
        _savefig(os.path.join(outdir, f"{name}.png"))

    shift_abs = shift.abs().stack().reset_index()
    shift_abs.columns = ["var1","var2","abs_shift"]
    shift_abs["pair"] = shift_abs["var1"] + " :: " + shift_abs["var2"]
    (shift_abs.sort_values("abs_shift", ascending=False)
              .to_csv(os.path.join(outdir, "corr_shift_long.csv"), index=False))

def backtest_masking_metrics(original, imputed, cols, outdir):
    """
    Computes simple point-error metrics only at true missing sites is complex
    (we don't know truth). As a proxy, compute MAE/RMSE between original and
    imputed data only on positions where the original dataset was observed, 
    and report distributional distances at missing sites (KS, means).
    """
    _ensure_dir(outdir)
    rows = []
    for col in tqdm(cols, desc="Backtest metrics"):
        if not _is_numeric(imputed[col]):
            continue
        obs_mask = original[col].notna()
        if obs_mask.sum() < 5:
            continue
        mae = np.mean(np.abs(original.loc[obs_mask, col] - imputed.loc[obs_mask, col]))
        rmse = np.sqrt(np.mean((original.loc[obs_mask, col] - imputed.loc[obs_mask, col])**2))
        r2 = np.corrcoef(original.loc[obs_mask, col], imputed.loc[obs_mask, col])[0,1]
        miss_mask = ~obs_mask
        if miss_mask.sum() >= 5 and obs_mask.sum() >= 5:
            ks_stat, ks_p = stats.ks_2samp(
                original.loc[obs_mask, col].dropna(),
                imputed.loc[miss_mask, col].dropna(),
                method="asymp"
            )
        else:
            ks_stat, ks_p = np.nan, np.nan

        rows.append({
            "column": col,
            "mae_on_observed": mae,
            "rmse_on_observed": rmse,
            "r_on_observed": r2,
            "ks_stat_obs_vs_imputedMiss": ks_stat,
            "ks_p_obs_vs_imputedMiss": ks_p,
            "n_obs_sites": int(obs_mask.sum()),
            "n_miss_sites": int(miss_mask.sum())
        })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "backtest_observed_vs_imputed.csv"), index=False)

def plot_miceforest_traces_if_available(kernel_dir, outdir):
    """
    If the team uses miceforest and saved a Kernel (pickle) with save_all_iterations=True,
    we can load it and render built-in diagnostic plots.
    """
    if kernel_dir is None:
        return
    try:
        import miceforest as mf  # type: ignore
    except Exception:
        warnings.warn("miceforest not installed; skipping MICE trace plots.")
        return

    candidates = list(Path(kernel_dir).glob("*.pkl")) + list(Path(kernel_dir).glob("*.pickle"))
    if not candidates:
        warnings.warn(f"No miceforest kernel found in {kernel_dir}; skipping traces.")
        return

    _ensure_dir(outdir)
    try:
        kernel = mf.ImputationKernel.load(candidates[0])
        kernel.plot_imputed_distributions()
        _savefig(os.path.join(outdir, "miceforest_imputed_distributions.png"))
        kernel.plot_mean_convergence()
        _savefig(os.path.join(outdir, "miceforest_mean_convergence.png"))
    except Exception as e:
        warnings.warn(f"Failed to render miceforest traces: {e}")


# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["knn", "missforest", "regression", "mice"],
                   help="Which imputer produced the file (for labeling only).")
    p.add_argument("--original", required=True, help="Path to original CSV with NaNs.")
    p.add_argument("--imputed",  required=True, help="Path to imputed CSV.")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID columns to exclude from stats/plots.")
    p.add_argument("--outdir", required=True, help="Base output dir (will create subdir per method).")
    p.add_argument("--miceforest-log-dir", default=None,
                   help="Optional dir containing miceforest ImputationKernel pickle for trace plots.")
    args = p.parse_args()

    method = args.method.lower()
    outbase = Path(args.outdir) / method
    _ensure_dir(outbase)

    # Load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = pd.read_csv(args.original)
        imputed  = pd.read_csv(args.imputed)

    # --- Clean up feature name prefixes like AA_ - ZZ_ ---
    import re
    original.columns = [re.sub(r'^[A-Z]{2}_', '', col) for col in original.columns]
    imputed.columns  = [re.sub(r'^[A-Z]{2}_', '', col) for col in imputed.columns]

    # Align columns and drop ID columns for analysis space
    original, imputed = _match_columns(original, imputed)
    work_orig = _subset_nonid(original, args.id_cols)
    work_imp  = _subset_nonid(imputed,  args.id_cols)
    cols = list(work_imp.columns)

    # 1) Missingness patterns
    plot_missingness(work_orig, work_imp, outdir=outbase / "missingness")

    # 2) Distribution checks
    plot_density_observed_vs_imputed(work_orig, work_imp, cols, outdir=outbase / "distributions")
    plot_qq_and_ks(work_orig, work_imp, cols, outdir=outbase / "qq_ks")

    # 3) Relationship + correlation structure
    plot_relationships(work_orig, work_imp, cols, outdir=outbase / "relationships")
    plot_corr_shift(work_orig, work_imp, cols, outdir=outbase / "correlations")

    # 4) Back-test on observed entries
    backtest_masking_metrics(work_orig, work_imp, cols, outdir=outbase / "backtest")

    # 5) Optional MICE traces (only if provided)
    if method == "mice":
        plot_miceforest_traces_if_available(args.miceforest_log_dir, outdir=outbase / "mice_traces")

    # Summary
    meta = {
        "method": method,
        "original_csv": args.original,
        "imputed_csv": args.imputed,
        "id_cols": args.id_cols
    }
    pd.Series(meta).to_json(outbase / "run_metadata.json", indent=2)
    print(f"[OK] Diagnostics written to: {outbase}")

if __name__ == "__main__":
    main()
