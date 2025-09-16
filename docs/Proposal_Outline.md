# proposal

Proposal outline. This is an artifically generated proposal outline to have some form of documentation to track
and scope capstone proposal. Most docs may end up not being submitted and can be archived as project artifacts.

---

### Infant & Under-Five Mortality Risk Monitoring

Title Page
  - Project Title: A Modular, Reproducible Machine Learning Framework for Infant and Under-Five Mortality Risk 
  Prediction

  - Student Name(s): [Your names here]

  - Advisor(s): [Advisor names here]

---

### Introduction
Background Information

Infant mortality rate (IMR) and under-five mortality rate (U5MR) are globally recognized indicators of population 
health and healthcare system effectiveness. They are key metrics for tracking progress toward Sustainable 
Development Goals (SDG 3.2), which call for ending preventable deaths of newborns and children under five. Despite 
steady global declines, disparities remain across states and districts, highlighting the need for targeted, 
data-driven interventions.

Research Gap/Challenges

- Data quality and reproducibility gaps: Mortality statistics are often incomplete, inconsistent across states, 
and delayed in reporting. Few systems implement standardized, reproducible data pipelines.

- Multi-level modeling gaps: Many studies focus either nationally or locally; few systems are modular enough to 
support both state-level and district-level analysis.

- Feature integration gaps: Maternal care, healthcare service coverage, socio-economic, and environmental data are 
rarely unified into a feature store for systematic analysis.

- Model drift and early-warning gaps: Current approaches are retrospective and static. There is limited research on 
real-time drift detection, recalibration, and adaptive risk scoring.

- Decision-support gaps: Policymakers lack accessible, modular dashboards that provide calibrated predictions, drift 
alerts, and regional comparability.

Objectives

This project aims to:

- Design a modular, reproducible ML framework for IMR/U5MR analysis at both state and district levels.

- Develop a feature store and clean dataset pipeline (Bronze → Silver → Gold layers).

- Train reproducible baseline ML models (scikit-learn) in Capstone 1, ensuring deterministic pipelines and model export.

- Integrate Tableau dashboards for decision support, with modular drill-down from state to district.

- Extend in Capstone 2 with drift detection, recalibration, and advanced models to enable a robust early-warning system.


### Dataset
2.1 Type of Data

Tabular, multi-level data: each row represents a geographic unit (state or district) per reporting period. Features 
include health indicators (antenatal care coverage, immunization rates), socio-economic factors (literacy, income 
proxies, urban/rural), and environmental data (temperature, rainfall anomalies).

2.2 Origin / Source

- Public repositories (e.g., UNICEF, WHO, World Bank, national health surveys).

- Government datasets (state- or district-level health statistics).

- [Health Analytics dataset](https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey) - India 
Annual Health Survey (AHS). 2012 - 2013

2.3 Size and Distribution

- Number of states: ~[X].

- Number of districts: ~[Y].

- Time range: [years covered].

- Target variables: IMR (deaths under age 1 per 1,000 live births), U5MR (deaths under age 5 per 1,000).

- Note: Data likely imbalanced, with some states/districts showing higher rates than others.

2.4 Data Quality and Challenges

- Missing values across states/districts.

- Inconsistent measurement/reporting (rates vs raw counts).

- Possible outliers from under-reporting or sudden shocks (pandemics, migration).

2.5 Dataset Example / Screenshot

(A table snippet showing `State`, `District`, `Year`, `IMR`, `ANC_Coverage`, `Facility_Deliveries`, 
`Immunization`, `UrbanRural`.)


### Methodology
3.1 Data Understanding and Exploration

- Perform exploratory data analysis (EDA).

- Summarize IMR/U5MR distributions across time, state, and district.

- Visualize trends with line plots, histograms, and geospatial heatmaps.

- Identify challenges: missing data patterns, class imbalance, temporal gaps.

3.2 Data Preprocessing and Preparation

- Bronze → Silver → Gold pipeline. (maybe Low, Medium, High)

- Handle missing data (MICE, MissForest, or simple imputation as baseline).

- Normalize/standardize continuous variables; encode categorical variables (e.g., State, UrbanRural).

- Ensure temporal integrity (no leakage across folds).

3.3 Modeling Plan

Capstone 1 (Baselines)

- Logistic regression (L2-regularized).

- Poisson/Negative Binomial regression for mortality rates.

- Random Forest Classifier.

- Deterministic sklearn Pipelines (with random_state).

- Model export for reproducibility (ONNX etc).

- Consider other model builds.


Capstone 2 (Exploration)

- Advanced models: XGBoost, LightGBM, TabNet. Build from baseline models or design newer models.

- Ensemble approaches and calibration methods.

- Drift detection (PSI, KL divergence, ADWIN, residual monitoring).

- Recalibration or retraining triggers based on drift.

3.4 Model Training Strategy

- Temporal cross-validation (TimeSeriesSplit).

- Train/validation/test based on reporting periods.

- MLflow logging: dataset hash, feature version, parameters, metrics.

- Batch scoring into Gold.RiskScores.

3.5 Evaluation Plan

- Classification metrics: R^2, RSME, AUROC, PR-AUC, Recall@Top-K, etc (for high-risk states/districts).

- Calibration metrics: Brier score, Expected Calibration Error (ECE).

- Regression metrics: RMSE, MAE for mortality rates.

- Error analysis: states/districts with highest residuals or drift signals.

3.6 Plan for Capstone 2 (Next Semester)

- Expand models beyond baselines.

- Integrate real-time drift detection + recalibration.

- System hardening: latency <150ms p95 inference, dashboard refresh <3s.

- Regional applicability: test pipeline on another region/country dataset.


### Expected Outcomes
Applications

- Provide decision-makers with a modular dashboard to track IMR/U5MR at both state and district levels.

- Enable early-warning alerts when data drift or risk shifts emerge.

- Deliver a reproducible ML pipeline useful for regional/global health research.

Deliverables

- GitHub repository (code, documentation, Makefile).

- Reproducible ML baselines (Capstone 1).

- Feature store + dataset pipeline.

- Exportable ML models + batch scoring service. (ONNX etc)

- Tableau dashboard (state/district risk scores, calibration plots, drift alerts).

- Capstone 2: extended models, drift detection, recalibration system, final report/poster.

---

