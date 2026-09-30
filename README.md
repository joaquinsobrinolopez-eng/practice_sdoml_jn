---
title: Practice Sdoml Demo
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---


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

## Execution & usage instructions
### Data exploration
Inspect the exploration notebook with feature distributions and data observations:
```bash
jupyter lab notebooks/1_exploration.ipynb
```

### Model training
Run the training loop and monitor the loss decrease:
```bash
uv run python -m practice_sdoml.modeling.train
```

### Performance evaluation & figures
Compute evaluation metrics and generate report artifacts in `reports/figures/`:
```bash
uv run python -m practice_sdoml.modeling.evaluate
```

## Documentation
Sphinx HTML documentation is located in `docs/build/html/`.

To recompile the documentation:
```bash
cd docs
uv run sphinx-build -b html source build/html
```

## Branch management & collaboration guides
Contributions follow the standard feature branch workflow:

1. Create a feature branch from: `main`: `git checkout -b feature/<feature-name>`.
2. Make meaningful commits with clear messages.
3. Merge back into `main` using explicit merge commits or pull requests.

## License
This project is licensed under the MIT License - see LICENSE file for details

##  Interactive Demonstration & Deployment 

This project is deployed and published as a modular package on **PyPI**: [`practice-sdoml-joaquin`](https://pypi.org/project/practice-sdoml-joaquin/).

### Launching the Demo

Anyone can launch the interactive Gradio demo immediately with zero configuration:

```bash
# Direct execution via uvx (from PyPI)
uvx --refresh practice-sdoml-joaquin
```

Alternatively, run from a local checkout:
```bash
# Sync environment and run locally
uv sync
uv run python practice_sdoml/app.py
```

The application provides: 

1. **Data Exploration:** Statistical summaries and interactive distribution plots of the dataset.
2. **Training Interface:** Adjustable hyperparameters (epochs, learning rate, batch size) with live progress tracking (`gr.Progress()`) and loss evolution curves.
3. **Model Evaluation:** Diagnostic tools including confusion matrices, calibration curves, and highest-loss sample analysis.

### Running the Demo Locally

Ensure dependencies are installed and run the application via `uv`:

```bash
# Option 1: Direct execution via uv
uv run python app.py

# Option 2: Using the CLI entry point
uv run practice-demo
```

## Members of the group
Joaquín Sobrino López
Nemer Awwad

## Project Organization

```
├── .github
│   └── workflows
│       └── deploy-docs.yml     <- CI/CD pipeline to automatically build and deploy Sphinx docs to GitHub Pages
├── .gitignore                  <- Excludes bytecode, virtual environments (.venv), data, and checkpoints
├── LICENSE                     <- Project open-source license (MIT)
├── Makefile                    <- Convenience commands for environment setup, training, and building docs
├── README.md                   <- Comprehensive project documentation, installation steps, and usage guides
├── pyproject.toml              <- Project metadata, dependencies, build settings (flit), and CLI entry points
├── uv.lock                     <- Pinned dependency lockfile managed by uv for guaranteed reproducibility
├── setup.cfg                   <- Configuration file for code linting tools (e.g., flake8)
│
├── data
│   ├── external                <- Data from third-party sources
│   ├── interim                 <- Intermediate transformed data
│   ├── processed               <- Final canonical datasets for modeling
│   └── raw
│       └── diabetes_risk.csv   <- Original, immutable dataset dump
│
├── docs
│   ├── Makefile                <- Build script for Sphinx documentation
│   ├── make.bat                <- Windows batch file for Sphinx commands
│   ├── source                  <- Sphinx source files, API declarations, and configuration
│   │   ├── conf.py             <- Sphinx configuration file
│   │   ├── index.rst           <- Root documentation index and table of contents
│   │   ├── modules.rst         <- Auto-generated module reference index
│   │   ├── practice_sdoml.rst  <- Package autodoc specification
│   │   └── practice_sdoml.modeling.rst <- Modeling subpackage autodoc specification
│   └── build                   <- Compiled static HTML documentation (excluded from version control)
│
├── models                      <- Serialized model weights, PyTorch checkpoints (.pt/.pth), or exports
│
├── notebooks
│   ├── 1_exploration.ipynb     <- Exploratory data analysis (EDA) with feature distributions and plots
│   └── 2_evaluation.ipynb      <- Interactive evaluation notebook
│
├── references                  <- Data dictionaries, documentation guides, and reference material
│
├── reports
│   ├── 1_exploration.html      <- Exported analysis reports
│   └── figures                 <- Generated graphic figures exported for reporting
│       ├── calibration_curve.png <- Reliability calibration diagram
│       ├── confusion_matrix.png  <- Multi-class / binary confusion matrix
│       └── top_loss_samples.png  <- Visualization of samples with the highest prediction error
│
└── practice_sdoml              <- Core source code package
    ├── __init__.py             <- Package initializer exposing key classes (DiabetesDataset, SimpleNet, train)
    ├── app.py                  <- Interactive Gradio application (Data Exploration, Training, Evaluation)
    ├── config.py               <- Project path definitions and environment configuration (loguru, dotenv)
    ├── dataset.py              <- Custom PyTorch DiabetesDataset class and get_dataloader utilities
    ├── features.py             <- Feature engineering and tabular preprocessing logic
    ├── plots.py                <- Standalone plotting functions (confusion matrix, calibration, top losses)
    └── modeling
        ├── __init__.py         <- Modeling module initializer
        ├── model.py            <- Neural network architecture definition (SimpleNet)
        ├── predict.py          <- Model inference logic for new observations
        ├── train.py            <- PyTorch training loop with loss reporting and optimization
        └── evaluate.py         <- Evaluation pipeline that computes metrics and saves report figures
```

--------

