# robustgp

[Robust Gaussian Process Regression Based on Iterative Trimming (ITGP)](https://arxiv.org/abs/2011.11057)

The Gaussian process (GP) regression can be severely biased when the data are contaminated by outliers. ITGP is a new robust GP regression algorithm that iteratively trims the most extreme data points. While the new algorithm retains the attractive properties of the standard GP as a nonparametric and flexible regression method, it can greatly improve the model accuracy for contaminated data even in the presence of extreme or abundant outliers. It is also easier to implement compared with previous robust GP variants that rely on approximate inference. Applied to a wide range of experiments with different contamination levels, the proposed method significantly outperforms the standard GP and the popular robust GP variant with the Student-t likelihood in most test cases.


## Install

```
pip install robustgp
```

Dependency: [GPy](https://github.com/SheffieldML/GPy/)


## Quick start

One can start with examples in [this notebook](https://nbviewer.jupyter.org/github/syrte/robustgp/blob/master/notebook/Example_Neal_Dataset.ipynb).


## Usage

```python
from robustgp import ITGP

# train ITGP
res = ITGP(X, Y, alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1)
gp, consistency = res.gp, res.consistency

# make prediction
y_avg, y_var = gp.predict(x_new)
y_var *= consistency
```
Here `gp` is a `GPy.core.GP` object, whose usage is further illustrated [here](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb).

- See the [docstring of ITGP](https://github.com/syrte/robustgp/blob/master/robustgp/robustgp.py) for detailed usage and full API.

- See this [notebook](https://github.com/syrte/robustgp/blob/master/notebook/Example_Neal_Dataset.ipynb) for a complete example.

## Benchmark

Comparison with standard Gaussian process and t-likelihood Gaussian process in terms of RMSE.
See the algorithm paper for details.

![Example sample](https://github.com/syrte/robustgp/blob/master/notebook/figure/neal_mock.png)

![Benchmark](https://github.com/syrte/robustgp/blob/master/notebook/figure/neal_bench.png)


## References

- Algorithm paper:
  [Robust Gaussian Process Regression Based on Iterative Trimming](https://arxiv.org/abs/2011.11057),
  Zhao-Zhou Li, Lu Li, & Zhengyi Shao, 2020

- First application:
  [Modeling Unresolved Binaries of Open Clusters in the Color-Magnitude Diagram. I. Method and Application of NGC 3532](https://ui.adsabs.harvard.edu/abs/2020ApJ...901...49L/),

  Lu Li, Zhengyi Shao, Zhao-Zhou Li, Jincheng Yu, Jing Zhong, & Li Chen, 2020

## License

The MIT License
