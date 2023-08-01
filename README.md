# PyCLA

A Python Package for Portfolio Optimization using the Critical Line Algorithm

![Tests](http://github.com/phschiele/PyCLA/workflows/Tests/badge.svg?event=push)

## Getting started

To use PyCLA, clone the repo and install the required dependencies.

```bash
git clone https://github.com/phschiele/PyCLA
```

Dependency installation using poetry:

```bash
poetry install
```

Dependency installation using pip:

```bash
pip install -r requirements.txt
```

## Using PyCLA - An Example

```py
import numpy as np

from pycla import PyCLA

n_sec = 10

# Expected returns and covariance matrix
np.random.seed(1)
mu = np.random.random(n_sec)
random_mat = np.random.rand(n_sec, n_sec)
C = np.dot(random_mat, random_mat.transpose())

# Lower and upper bounds
lb = np.ones(n_sec) * 0.05
ub = np.ones(n_sec) * 0.7

# Equality constraints
A = np.ones((1, n_sec))
b = np.array([1])

# Inequality constraints
first_n = 5  # combined weight of first five assets <= 50%
A_in = np.array([[1]*first_n + [0]*(n_sec-first_n)])
b_in = np.array([0.5])

# Create the PyCLA object and trace the frontier
pycla = PyCLA(mu, C, A, b, A_in, b_in, lb, ub)
pycla.trace_frontier()

```
