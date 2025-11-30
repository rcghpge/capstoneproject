# capstoneproject

<p align="center">
  <strong>Capstone Project 1 | Division of Data Science | The University of Texas at Arlington</strong>

<p align="left">
  Machine learning utilizing key health indicators for infant mortality rate prediction.<br>
  <strong>References -</strong> Kaggle. Health Analytics. India. Annual Health Survey (AHS): https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey
</p>

<br>

---

# Capstone Project 1 Structure
```bash
.
├── models
├── notebooks
├── submissions
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── __init__.py
└── bandit.yml

3 directories, 7 files
```

---

## Getting Started
Clone the GitHub repository and generate a Python virtual environment. Install required software dependencies.
Runs in Jupyter Notebook, Jupyter Lab, and Bash command-line interpreter environments.

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

# Launch Jupyter Notebooks or Jupyter Lab
jupyter lab

# Bash Command-line - CLI
# Secure Python code scanning with Bandit and pip-audit
bandit -r . -c bandit.yml --severity-level high --confidence-level high

pip-audit
```

---

License: MIT

---
