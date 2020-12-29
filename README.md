# robustgp
Robust Gaussian Process Regression Based on Iterative Trimming
https://arxiv.org/abs/2011.11057

First application
- Modeling Unresolved Binaries of Open Clusters in the Color-Magnitude Diagram. I. Method and Application of NGC 3532
  https://ui.adsabs.harvard.edu/abs/2020ApJ...901...49L/abstract

## Dependency
[GPy](https://github.com/SheffieldML/GPy/)

## Install

Install the dependency first
```
pip install GPy
```

Assuming you want to put the code at certain directory, say `~/pythonlib`
```
cd ~/pythonlib
wget https://raw.githubusercontent.com/syrte/robustgp/master/robustgp/robustgp.py
```

Then you can import `ITGP` as following,
```
import sys
sys.path.append('~/pythonlib')
from robustgp import ITGP
```

Or add this in your `.bashrc` once for all
```
export PYTHONPATH="$HOME/pythonlib:$PYTHONPATH"
```

I will write a `setup.py` in the future for easier installation.


## Usage

You have two 1d arrays, `X` and `Y`, and want to predict the value of `Y` at arbitrary `X`.

```python
import numpy as np
from robustgp import ITGP
import GPy

k_sigma2 = 0.1**2
k_length = 2
w_sigma2 = 0.001**2

kernel_rbf = GPy.kern.RBF(input_dim=1, variance=k_sigma2, lengthscale=k_length)
kernel_mat32 = GPy.kern.Matern32(input_dim=1, variance=k_sigma2, lengthscale=k_length)
kernel_white = GPy.kern.White(input_dim=1, variance=w_sigma2)
kernel = kernel_mat32 + kernel_white

g_lik = GPy.likelihoods.Gaussian(variance=1e-10)
g_lik.variance.constrain_fixed(1e-10)

gp, consistency, score = ITGP(
    X.reshape(-1, 1), Y.reshape(-1, 1),
    alpha1=0.5, alpha2=0.95,
    niter0=10, niter1=10, niter2=0,
    kernel=kernel, likelihood=g_lik,
)
gp.optimize()

x_new = np.linspace(X.min(), X.max(), 100)
y_mean, y_var = gp.predict(x_new)

plt.plot(x_new, y_mean)
```

`gp` is a `GPy.core.GP` object. Please refer the usage of [GPy](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb).

You can start with, e.g., `gp.plot()`.

More detailed user guide and notebook example will be added later.

## License 
The MIT License
