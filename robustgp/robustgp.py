"""
Robust Gaussian Process Regression Based on Iterative Trimming
Zhao-Zhou Li, Lu Li, Zhengyi Shao, 2020
https://arxiv.org/abs/2011.11057


Change logs
-----------
May 04 2021
Updated for revised manuscript, with changes in API and default parameters

Aug 11 2020
Initial commit.
"""

import GPy
import numpy as np
from scipy.stats import chi2
from collections import namedtuple

__all__ = ['ITGP']
__version__ = 2.0
__author__ = "Zhaozhou Li"


def ITGP(X, Y, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1,
         maxiter=None, return_predict=True, callback=None, callback_args=(),
         warm_start=False, optimize_kwargs={}, **gp_kwargs):
    """
    Robust Gaussian Process Regression Based on Iterative Trimming.

    Parameters
    ----------
    X: array shape (n, d)
    Y: array shape (n, 1)
        Input data with shape (# of data, # of dims).
    alpha1, alpha2: float in (0, 1)
        Trimming and reweighting parameters respectively.
    nsh, ncc, nrw: int (>=0)
        Number of shrinking, concentrating, and reweighting iterations respectively.
    return_predict: bool
        If True, then the predicted mean, variance, and score of input data will be returned.
    callback: callable
        Function for monitoring the iteration process. It takes
        the iteration number i and the locals() dict as input
        e.g.
            callback=lambda i, locals: print(i, locals['gp'].num_data, locals['gp'].param_array)
        or
            callback=lambda i, locals: locals['gp'].plot()
    callback_args:
        Extra parameters for callback.
    warm_start: bool, int
        From which step it uses the warm start for optimizing hyper-parameters.
            0: (default) disable warm start, always use a fresh initial guess (provided by input gp object).
          >=1: start optimization with hyper-parameters trained from last iteration for steps >= warm_start,
        A warm start might help converge faster with the risk of being trapped at a local solution.
    optimize_kwargs:
        GPy.core.GP.optimize parameters.
    **gp_kwargs:
        GPy.core.GP parameters, including likelihood and kernel.
        Gaussian and RBF are used as defaults.

    Returns
    -------
    ITGPResult: named tuple object
        gp:
            GPy.core.GP object.
        consistency:
            Consistency factor.
        ix_sub:
            Boolean index for trimming sample.
        niter:
            Total iterations performed, <= 1 + nsh + ncc + nrw.
        Y_avg, Y_var:
            Expectation and variance of input data points. None if return_predict=False.
        score:
            Scaled residuals. None if return_predict=False.
    """
    # check parameters
    if X.ndim == 1:
        X = np.atleast_2d(X).T
    if Y.ndim == 1:
        Y = np.atleast_2d(Y).T
    if len(X) != len(Y):
        raise ValueError("X should have the same length as Y")

    n, p = Y.shape
    if p != 1:
        raise ValueError("Y is expected in shape (n, 1).")
    if n * alpha1 - 0.5 <= 2:
        raise ValueError("The dataset is unreasonably small!")

    if nsh < 0 or ncc < 0 or nrw < 0:
        raise ValueError("nsh, ncc and nrw are expected to be nonnegative numbers.")

    gp_kwargs.setdefault('likelihood', GPy.likelihoods.Gaussian(variance=1.0))
    gp_kwargs.setdefault('kernel', GPy.kern.RBF(X.shape[1]))
    gp_kwargs.setdefault('name', 'ITGP regression')

    # use copies so that input likelihood and kernel will not be changed
    likelihood_init = gp_kwargs['likelihood'].copy()
    kernel_init = gp_kwargs['kernel'].copy()

    # temp vars declaration
    d_sq = None
    ix_old = None
    niter = 0

    # shrinking and concentrating
    for i in range(1 + nsh + ncc):
        if i == 0:
            # starting with the full sample
            ix_sub = slice(None)
            consistency = 1.0
        else:
            # reducing alpha from 1 to alpha1 gradually
            if i <= nsh:
                alpha = alpha1 + (1 - alpha1) * (1 - i / (nsh + 1))
            else:
                alpha = alpha1
            chi_sq = chi2(p).ppf(alpha)
            h = int(min(np.ceil(n * alpha - 0.5), n - 1))  # alpha <= (h+0.5)/n

            # XXX: might be buggy when there are identical data points
            # better to use argpartition! but may break ix_sub == ix_old.
            ix_sub = (d_sq <= np.partition(d_sq, h)[h])  # alpha-quantile
            consistency = alpha / chi2(p + 2).cdf(chi_sq)

        # check convergence
        if (i > nsh + 1) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        # warm start?
        if 0 == warm_start or niter < warm_start:
            gp_kwargs['likelihood'] = likelihood_init.copy()
            gp_kwargs['kernel'] = kernel_init.copy()

        # train GP
        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **gp_kwargs)
        gp.optimize(**optimize_kwargs)

        # make prediction
        Y_avg, Y_var = gp.predict(X, include_likelihood=True)
        d_sq = ((Y - Y_avg)**2 / Y_var).ravel()

        if callback is not None:
            callback(niter, locals(), *callback_args)
        niter += 1

    # reweighting
    for i in range(nrw):
        alpha = alpha2
        chi_sq = chi2(p).ppf(alpha)

        # XXX: might be buggy when there are identical data points
        ix_sub = (d_sq <= chi_sq * consistency)
        consistency = alpha / chi2(p + 2).cdf(chi_sq)

        # check convergence
        if (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        # warm start?
        if 0 == warm_start or niter < warm_start:
            gp_kwargs['likelihood'] = likelihood_init.copy()
            gp_kwargs['kernel'] = kernel_init.copy()

        # train GP
        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **gp_kwargs)
        gp.optimize(**optimize_kwargs)

        # make prediction
        if i < nrw - 1 or return_predict:
            Y_avg, Y_var = gp.predict(X, include_likelihood=True)
            d_sq = ((Y - Y_avg)**2 / Y_var).ravel()
        else:
            pass  # skip final training unless prediction is wanted

        if callback is not None:
            callback(niter, locals(), *callback_args)
        niter += 1

    if return_predict:
        # outlier detection
        score = (d_sq / consistency)**0.5
        return ITGPResult(gp, consistency, ix_sub, niter, Y_avg, Y_var, score)
    else:
        return ITGPResult(gp, consistency, ix_sub, niter, None, None, None)


ITGPResult = namedtuple('ITGPResult', ('gp', 'consistency', 'ix_sub', 'niter', 'Y_avg', 'Y_var', 'score'))
