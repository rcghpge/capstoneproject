# capstoneproject

<p align="center">
  <strong>Capstone Project 1 | Division of Data Science | The University of Texas at Arlington</strong>
</p>
<table align="center" width="100%">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/UTA Celebrating 130 Years logo white circle.png" alt="UTA Logo"
           style="width: auto; height: auto;" />
  </tr>
</table>

<p>
  Machine learning utilizing key health indicators for infant mortality rate prediction.
</p>

<p>
  <strong>References</strong> 
  <br>
  Health Analytics. India. Annual Health Survey (AHS)
  <br>
  <strong>Kaggle:</strong> https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey
</p>

---

# Capstone Project 1 Structure
```bash
.
├── assets
├── models
├── notebooks
├── submissions
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── __init__.py
├── requirements.txt
└── pyproject.yml

4 directories, 8 files
```

---

## Getting Started
Clone the GitHub repository and generate a Python virtual environment. Install required software dependencies.
Runs in Jupyter Notebook, Jupyter Lab, and Bash command-line environments.

```bash
# Clone repository
git clone https://github.com/rcghpge/capstoneproject.git
cd capstoneproject

# Generate pip venv 
python -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies 
pip install -e .[dev]

# Environment Checks
python -c "from models import *; print('✅ Model import dependencies OK')"
bandit -r models/
bandit -r models/ -f json -o security-report.jon # secure report summary
pip-audit -r requirements.txt
pip check
pytest --cov=models/ --cov-report=term-missing

# Run Python models and Launch Jupyter for EDA 
jupyter lab notebooks/ # run in a web browser environment
jupyter lab notebooks/ --no-browser # intiliaze Jupyter server with no web browser
jupyter lab/models/
jupyter lab/models/ --no-browser
```

---

License: MIT

---
