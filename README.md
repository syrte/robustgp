# robustgp
Robust Gaussian Process Regression Based on Iterative Trimming
https://arxiv.org/abs/2011.11057

First application
- Modeling Unresolved Binaries of Open Clusters in the Color-Magnitude Diagram. I. Method and Application of NGC 3532
  https://ui.adsabs.harvard.edu/abs/2020ApJ...901...49L/abstract

## Dependency
[GPy](https://github.com/SheffieldML/GPy/)

## Usage
```
from robustgp import ITGP
gp, consistency, score = ITGP(X, Y)
```
`gp` is a GPy.core.GP object. Please refer the usage of GPy.

You can start with, e.g., `gp.plot()`.

More detailed user guide will be added later.

## License 
The MIT License
