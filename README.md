# Dark Grey Box

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CircleCI](https://circleci.com/gh/czagoni/darkgreybox.svg?style=shield)](https://circleci.com/gh/czagoni/darkgreybox)
[![PyPI version](https://badge.fury.io/py/darkgreybox.svg)](https://badge.fury.io/py/darkgreybox)

## DarkGreyBox: An open-source data-driven python building thermal model inspired by Genetic Algorithms and Machine Learning

### Read the medium article here: https://medium.com/analytics-vidhya/data-driven-thermal-models-for-buildings-15385f744fc5

Constructing simple, accurate and easy-to-interpret thermal models for existing buildings is essential in reducing the environmental impact of our built environment. DarkGreyBox provides a data-driven approach to constructing and fitting RC-equivalent grey box thermal models for buildings, within the classic Machine Learning (ML) framework for straightforward model performance evaluation. 

A large number of competing models can be set up in easy-to-configure pipelines and the best performing models are selected based on principles inspired by Genetic Algorithms (GA). This approach also addresses the main disadvanatages of classical grey-box thermal modelling techniques by not requiring initial condition values for the thermal parameters to be pre-calculated and also not requiring an excitation signal to be injected into the building for successful model convergence and evaluation.
 
The massive advantages of using a DarkGreyBoxModel over a black-box (i.e. Machine Learning) model - e.g. a deep sequence-to-sequence model - are that it is easily interpreted by humans and that it slots easily into other modelling frameworks. E.g. to model the behaviour of a building with its connected heating system, simply construct a heat source model in a MILP framework and the grey-box building thermal model just slots in as a set of linear differential equations with a handful of parameters. Doing this with a deep ML model would be quite tricky. 

The easiest way to get familiar with DarkGreyBox is to look at the [tutorials](docs/tutorials/).

## Installation

### Dependencies

DarkGreyBox requires:

- Python (>= 3.6)
- lmfit (>= 1.0.2)
- pandas (>= 1.2.3)
- joblib (>= 1.0.1)

Note: these are only the core dependencies and you will most likely want to install either the optional dependencies or your preferred custom alternatives to them.

### User installation from PyPi package (latest release)

Install DarkGreyBox via `pip`:
```bash
pip install darkgreybox
```

#### Optional Dependencies

This gives you a headstart for using DarkGreyBox in anger.

- scikit-learn (>=0.24.1)
- numdifftools (>=0.9.39)
- statsmodels (>=0.12.1)
- matplotlib (>=3.4.0)
- jupyter (>=1.0.0)
- notebook (>=6.1.5)

You can install these additional dependencies via pip:
```bash
pip install darkgreybox[dev]
```

### User installation from source repository

You can check the latest sources with the command::
```bash
git clone https://github.com/czagoni/darkgreybox.git
```

You can install the dev dependencies (from the root of the repository):
```bash
pip install -e .'[dev]'
```

## Documentation

To access the tutorials you need to have cloned DarkGreyBox from the source repository (see above).

### Tutorials

The easiest way to get into the details of how DarkGreyBox works is through following the tutorials:

* [Demo Notebook 01 - Ti Model Direct Fit](docs/tutorials/darkgrey_poc_demo_01.ipynb): This notebook demonstrates the direct usage of the DarkGreyBox models via a simple fitting example for a Ti model.
* [Demo Notebook 02 - TiTe Model Direct Fit FAIL](docs/tutorials/darkgrey_poc_demo_02.ipynb): This notebook demonstrates the direct usage of the DarkGreyBox models via a simple fitting example for a TiTe model. In this case a local minimum is found during the fitting process and the model heavily oscillates making it unusable.
* [Demo Notebook 03 - TiTe Model Wrapper Fit PASS](docs/tutorials/darkgrey_poc_demo_03.ipynb): This notebook demonstrates the usage of the DarkGreyBox models via fitting them with a wrapper function for a TiTe model.
* [Demo Notebook 04 - DarkGreyFit](docs/tutorials/darkgrey_poc_demo_04.ipynb): This notebook demonstrates the usage of the DarkGreyBox models via fitting them with DarkGreyFit, setting up and evaluating multiple pipelines at once.

Launch a new jupyter notebook from the root of the repository:
```bash
jupyter notebook
```

## Development

We welcome new contributors of all experience levels. 

### Source code

You can check the latest sources with the command::
```bash
git clone https://github.com/czagoni/darkgreybox.git
```

You can install the dev and test dependencies (from the root of the repository):
```bash
pip install -e .'[dev,test]'
```

### Testing

After installation, you can launch the test suite from the repo root
directory (you will need to have `pytest` >= 6.2.2 installed):
```bash
pytest
```

You can check linting from the repo root directory (you will need to have `flake8` >= 3.9.0 installed):
```bash
flake8
```


