# roadmap

Capstone roadmap. Not a strict roadmap but project documentation to track progress. 

---

### Research Question

How can we design a reproducible, modular, real-time system that integrates a feature store, baseline machine learning 
models, and drift detection to provide accurate, calibrated, and actionable risk scoring of infant and under-five mortality rates (IMR/U5MR) at both the state and district levels, while addressing gaps in data quality, 
interpretability, and early warning for public health decision-making across regions? 

### Problem Scope (State + Regional Modularity)

Infant mortality rate (IMR) and under-five mortality rate (U5MR) are critical population health indicators used to 
track Sustainable Development Goals (SDGs). Despite significant progress, research gaps remain:

- Data quality and reproducibility gaps: Mortality data is often incomplete, inconsistent, and not harmonized across 
states. Current models are not always reproducible due to ad hoc preprocessing and lack of standardized pipelines.

- Multi-level modeling gaps: Most studies focus either nationally or locally; few systems allow modular scalability 
across state-level and district-level units, hindering regional comparability.

- Feature integration gaps: State- and district-level determinants (maternal health, service coverage, socio-economic 
and environmental factors) are rarely unified into a modular feature store for reuse across different geographies.

- Model drift and early warning gaps: Current approaches are mostly retrospective; few incorporate real-time drift 
detection and adaptive recalibration, which are essential for detecting policy impacts or emerging risks.

- Decision-support (reserach gaps): Policymakers at both state and regional levels need modular dashboards that can 
pivot between scales (district ↔ state ↔ region), but most existing tools are static or non-reproducible.

This project looks to address these research gaps by building a modular, state- and district-level risk scoring system 
that is reproducible, regionally extensible, and designed for scalable research applications.

---

### Roadmap (Capstone 1 + Capstone 2)
Capstone 1 (Foundations: Baselines + Modularity)

- Data ingestion & curation: Gather IMR/U5MR, maternal/child health, socio-economic, and environmental datasets. 
Normalize into Bronze → Silver → Gold lakehouse (database) layers.

- Feature store (modular v1): Define state- and district-level feature views in YAML or other method (reusable 
across regions).

- Baseline ML models (scikit-learn): Logistic regression, Poisson/negative binomial regression, Random Forest.

- Reproducibility: Locked environments, deterministic sklearn pipelines, MLflow logging, ONNX export.

- Batch scoring: Generate Gold.RiskScores at both district and state levels.

- Visualization: Build initial Tableau dashboards that support multi-level exploration (state heatmaps, district drill-downs).

- Deliverable: Reproducible modular framework + working Tableau dashboard at state and district levels.


Capstone 2 (Exploration: System Hardening + Regional Applicability)

- Model exploration: Extend baseline suite with XGBoost, LightGBM, calibrated ensembles.

- Drift detection & early warning: Implement PSI, KL divergence, ADWIN, and residual monitoring at both scales (state & district).

- Adaptive recalibration: Implement recalibration or retrain triggers based on detected drift.

- Operational metrics:

  - API inference latency <150ms (p95)

  - Tableau refresh <3s

  - Drift detection delay <2 reporting periods

- Regional modularity: Validate system design for portability (can adapt to another country/region with minimal changes).

- Final deliverable: 
  - GitHub repository (code + docs), reproducible sklearn baselines (Capstone 1), ONNX models, Tableau dashboard, monitoring 
  system (Capstone 2), PowerPoint presentation, proposals, final research report/poster.

  - A modular, state- and district-level early-warning system with reproducible ML, drift monitoring, 
  and a multi-level Tableau dashboard.

---
