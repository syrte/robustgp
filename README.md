# robustgp
Robust Gaussian Process Regression Based on Iterative Trimming
Zhao-Zhou Li, Lu Li, Zhengyi Shao, 2020
https://arxiv.org/abs/2011.11057

First application
- Modeling Unresolved Binaries of Open Clusters in the Color-Magnitude Diagram. I. Method and Application of NGC 3532
  https://ui.adsabs.harvard.edu/abs/2020ApJ...901...49L/

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

```python
from robustgp import ITGP

# train ITGP
res = ITGP(X, Y, alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1)
gp, consistency = res.gp, res.consistency

# make prediction
y_avg, y_var = gp.predict(x_new)
y_var *= consistency
```

`gp` is a `GPy.core.GP` object. Please refer the usage of [GPy](https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb).

See [this notebook](https://github.com/syrte/robustgp/blob/master/notebook/Example_Neal_Dataset.ipynb) for a complete example.

## License 
The MIT License
