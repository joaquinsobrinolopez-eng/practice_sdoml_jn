# practice_SDOML

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## Documentation website
The complete HTML documentation is built and published automatically via GitHub Actions:
**[View Documentation Online](https://joaquinsobrinolopez-eng.github.io/practice_sdoml_jn/)**

## Description of the project
This project implements a Machine Learning Pipeline for data classification, using PyTorch. It
trains a neural network (`SimpleNet`) on the `diabetes_risk.csv` dataset. Also, while it is 
training, it tracks progeression and generate exploratory and evaluation metrics.

## Data source
It is `diabetes_risk.csv`, which it has been downloaded from Kaggle, a public Machine Learning
Repository. It is locally stored in `data/raw/diabetes_risk.csv`

## Installation & Environment Setup 
```
This project uses `uv` for lightning-fast dependency management and reproducibility.

1. Install `uv` (if not already installed):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Sync the environment and install dependencies:
   `uv sync`
3. Run the training script:
   `uv run python -m practice_sdoml.modeling.train`
```
## Documentation
Sphinx HTML documentation is located in `docs/build/html/`.

To recompile the documentation:
```bash
cd docs
uv run sphinx-build -b html source build/html
```


## Members of the group
Nemer Awwad
Joaquín Sobrino López

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump (diabetes_risk.csv)
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│   └── 1_exploration.ipynb   <- Jupyter notebook in which data exploration is done
│   └── 2_evaluation.ipynb   <- Jupyter notebook in which data evaluation is done
|
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         practice_sdoml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── practice_sdoml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes practice_sdoml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── model.py            <- Code in which the neural network for the prediction is showed
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

