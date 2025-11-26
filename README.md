# capstoneproject

Capstone Project 1 | Division of Data Science | The University of Texas at Arlington

Machine learning utilizing key health indicators to assess infant mortality

References:
Kaggle
Health Analytics - India. Annual Health Survey (AHS): [link](https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey) 

---

# Capstone Project 1 Structure
```bash
.
├── CITATION.cff
├── LICENSE
├── README.md
├── __init__.py
├── bandit.yml
├── bandit.yml~
├── data
├── docs
├── mice_imputer2.patch
├── models
├── notebooks
├── out
├── prototype
├── submissions
├── test
├── tools
└── venv

11 directories, 7 files
```

---

## Getting Started
Clone the GitHub repository and generate a Python virtual environment. Install require software dependencies.
For Jupyter Notebook, Jupyter Lab, and Bash command-line interpreter environments.

```bash
# Clone repository
git clone https://github.com/rcghpge/capstoneproject.git
cd capstoneproject

# Generate pip venv if needed
python -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies and Jupyter Notebooks or Jupyter Lab
pip install jupyterlab jupyter notebook # add required dependencies as needed - pandas numpy matplotlib etc.

# Launch Jupyter Notebook or Jupyter Lab
jupyter lab

# CLI
Secure Python code scanning with Bandit and pip-audit
bandit -r . -c bandit.yml --severity-level high --confidence-level high

pip-audit
```

---

License: MIT

---
