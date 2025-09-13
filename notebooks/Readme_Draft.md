# notebooks

## Methodologies for Capstone
This is a list of preprocessing methodologies available for data preprocessing respective to the dataset utilized in this body of work.

---

## Overview/General Summary:
This document summarizes state-of-the-art traditional ML methodologies for preprocessing and feature engineering in high-dimensional datasets (where the number of features ≫ number of rows). These techniques help ensure stability, interpretability, and reproducibility across imputation, feature selection, dimensionality reduction, and regularization.

### Missing data (imputation)
- MICE / Iterative Imputer (linear or tree-based): strong baseline; do it inside CV folds.
- MissForest / miceforest (random-forest/LightGBM-based iterative imputation): often stronger than linear MICE on mixed/tabular data; handles nonlinearity well.
- SoftImpute / matrix completion: good when data are approximately low-rank (works best with many correlated features).
- KNN imputer: simple, but can be brittle in p≫n; use with care and scaling.

### Feature selection (avoid overfitting, improve stability)
- Filter methods (fast screeners)
- Univariate tests with FDR control (Benjamini–Hochberg).
- Mutual Information, mRMR, ReliefF for nonlinearity.
- Sure Independence Screening (SIS) as a first pass when p is huge.
- Embedded methods (preferable in p≫n)
- L1/Elastic Net (LASSO, EN): canonical for sparse, stable selection.
- Tree-based (Random Forest/GBDT) importances with permutation importance; pair with Boruta or Boruta-SHAP for conservative selection.
- Stability Selection (bootstrapped LASSO/EN with selection probability thresholds): excellent for reproducibility.
- Wrapper methods
- RFE with a strongly regularized base model; couple with nested CV + stability checks.

### Dimension reduction (when selection alone isn’t enough)
- Sparse PCA / Truncated SVD (for sparse/high-dim data).
- PLS/PLS-DA for supervised low-rank projections when targets guide compression.
- Supervised PCA (project onto PCs correlated with y).
- (t-SNE/UMAP for visualization only, not features into the final model.)

### Feature construction/engineering (compact, controlled)
- Polynomial/interaction terms with L1/EN (let the penalty pick).
- Group/Hierarchical penalties (Group LASSO, Sparse-Group LASSO) if features come in known groups (e.g., dummies of a categorical).
- Target/impact encoding for high-cardinality categoricals, but strictly cross-fitted to prevent leakage.

### Regularization & modeling methodologies that carry p≫n (predictors (features) >> mumber of samples (rows))
- Linear models with heavy shrinkage: Ridge / EN logistic/linear, calibrated if needed.
- Kernel methods with strong regularization (linear/low-rank kernels).
- Tree ensembles (RF/GBDT) with aggressive early-stopping and permutation importance for selection—not just raw gain.
- Calibrate (Platt/isotonic) after selection if probabilities matter.

### Robust training protocol
- Nested CV (outer for performance, inner for tuning/selection/imputation).
- All preprocessing inside the pipeline (impute → scale → select → reduce → model).
- Stability audits: bootstrap/cross-fit to get selection frequencies; report them.
- Permutation tests or knockoff-style controls when multiple testing risk is high.
- Leakage checks: encoders, imputers, scalers, and selectors must be fit on train folds only.

### Practical Python building blocks
- Imputation: `sklearn.IterativeImputer`, miceforest, missingpy.MissForest, fancyimpute.SoftImpute
- Selection: `sklearn.feature_selection` (FDR, MI), BorutaPy, stability-selection (e.g., `skglm`/`scikit-learn-contrib`), `RFECV`
- Reduction: `SparsePCA`, `TruncatedSVD`, `PLSRegression`
- Encoding: `category_encoders` (Target/WOE/LeaveOneOut with CV)

### Example template p≫n pipeline
- Column typing & leakage plan (what’s known at train time).
- Impute (miceforest or IterativeImputer) per fold.
- Screen with filters + FDR (e.g., keep top k by MI within FDR q).
- Embedded selection with EN or Boruta; record stability frequency over bootstraps.
- Optional SparsePCA/PLS to a small k.
- Final model: Ridge/EN or calibrated GBDT.
- Evaluate via nested CV; report feature stability and permutation importance.

### How MICE fits
- Keep MICE (or miceforest) as your fold-internal imputer, but pair it with:
- Stability-selection EN (sparser, more reproducible feature set).
- Or Boruta-SHAP on a small, leakage-proof GBDT.
- Optionally SparsePCA/PLS when features are highly collinear.

---

### Imputation methods (filling missing values)
- MICE (Multiple Imputation by Chained Equations) → Iteratively models each feature with missing values as a function of other features.
- MissForest → A random forest–based iterative imputer. Handles mixed data types (categorical + numeric), nonlinear relationships.
- miceforest → A fast, Python implementation of MICE that uses LightGBM instead of linear models.
- SoftImpute / matrix completion → Assumes the data matrix is approximately low-rank; fills in missing values by solving a low-rank matrix factorization problem. Works well when variables are highly correlated.
- KNN Imputer → Replaces missing values with averages (or votes) from the k nearest neighbors. Sensitive to scaling and distance metric.

### Feature selection & importance measures
- FDR (False Discovery Rate) → A statistical correction method (e.g., Benjamini–Hochberg) that controls the expected proportion of false positives among selected features.
- MI (Mutual Information) → A measure of how much knowing one variable reduces uncertainty about another. Nonlinear, captures relationships missed by correlation.
- mRMR (Minimum Redundancy Maximum Relevance) → Feature selection method: keeps features that are highly relevant to the target (max relevance) while being minimally correlated with each other (min redundancy).
- Boruta → A wrapper feature selection method around Random Forest. Compares the importance of real features to “shadow” features (randomized copies) and only keeps features that consistently beat the shadows.
- SIS (Sure Independence Screening) → Very high-dimensional filter method: quickly screens out irrelevant features based on simple correlations (or similar measures) before applying more advanced modeling.

### Dimension reduction methods
- SparsePCA (Sparse Principal Component Analysis) → Variant of PCA that enforces sparsity in the components (so each component depends only on a few features). Improves interpretability in high dimensions.
- Truncated SVD (Singular Value Decomposition) → Similar to PCA but works directly on sparse matrices (like TF-IDF or bag-of-words) without centering the data. Often used in text mining and recommendation systems.
- PLS (Partial Least Squares) → Supervised dimension reduction: finds components that explain covariance between features and target.
- PLS-DA (Partial Least Squares – Discriminant Analysis) → PLS adapted for classification problems (supervised, categorical target).

### Regularization methods
- Ridge → Linear model with L2 penalty (shrinks coefficients toward zero, but rarely sets them exactly to zero). Good for multicollinearity.
- L1 (Lasso) → Linear model with L1 penalty (forces many coefficients to be exactly zero). Does both regularization and feature selection.
- EN (Elastic Net) → Combines L1 + L2 penalties. Balances sparsity (L1) and stability (L2). Often the best default for p≫n problems.

### Feature engineering general summary:
- Imputation tools: MissForest, miceforest, SoftImpute, KNN Imputer.
- Feature selection tools: FDR, MI, mRMR, Boruta, SIS.
- Dimension reduction tools: SparsePCA, Truncated SVD, PLS/PLS-DA.
- Regularization tools: Ridge, Lasso (L1), Elastic Net (EN).
- Imputation tools: MissForest, miceforest, SoftImpute, KNN Imputer.
- Feature selection tools: FDR, MI, mRMR, Boruta, SIS.
- Dimension reduction tools: SparsePCA, Truncated SVD, PLS/PLS-DA.
- Regularization tools: Ridge, Lasso (L1), Elastic Net (EN).

---


