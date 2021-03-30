import GPy
import numpy as np
from scipy.stats import chi2
from collections import namedtuple

__all__ = ['ITGP']


def ITGP(X, Y, alpha1=0.50, alpha2=0.95,
         niter0=5, niter1=10, niter2=0,
         callback=None, callback_args=(), debug=False,
         **kwargs):
    """
    Robust Gaussian Process Regression Based on Iterative Trimming.

    Parameters
    ----------
    X: array shape (n, p)
    Y: array shape (n, 1)
        Input data.
    alpha1, alpha2:
        Coverage fraction used in contraction step and reweighting step respectively.
    niter0:
        Number of shrinking iteration.
    niter1, niter2:
        Maximum iteration allowed in contraction step and reweighting step respectively.
    callback: callable
        Function for checking the iteration process. It takes
        the GPRegression object `gp`, consistency factor and iteration number `i` as input
        e.g.
            callback=lambda gp, c, i: print(i, gp.num_data, gp.param_array)
        or
            callback=lambda gp, c, i: gp.plot()
    callback_args:
        Extra parameters for callback.
    **kwargs:
        GPy.core.GP parameters.

    Returns
    -------
    gp:
        GPy.core.GP object.
    consistency:
        Consistency factor.
    ix_out:
        Boolean index for outliers.
    """
    n, p = Y.shape
    if p != 1:
        raise ValueError("Y is expected in shape (n, 1).")

    kwargs.setdefault('likelihood', GPy.likelihoods.Gaussian(variance=1.0))
    kwargs.setdefault('kernel', GPy.kern.RBF(X.shape[1]))
    kwargs.setdefault('name', 'Robust GP regression')

    # first iteration
    gp = GPy.core.GP(X, Y, **kwargs)
    gp.optimize()
    consistency = 1
    mean, var = gp.predict(X)
    dist = (Y - mean)**2 / var

    iter_num = 0
    if callback is not None:
        callback(gp, consistency, iter_num, *callback_args)

    ix_old = None
    niter1 = niter0 + niter1

    # contraction step
    for i in range(niter1):
        if i < niter0:
            # reduce alpha_ from 1 to alpha1 gradually
            alpha_ = alpha1 + (1 - alpha1) * ((niter0 - 1 - i) / niter0)
        else:
            alpha_ = alpha1
        h = min(int(np.ceil(n * alpha_)), n) - 1
        dist_th = np.partition(dist.ravel(), h)[h]
        eta_sq1 = chi2(p).ppf(alpha_)
        ix_sub = (dist <= dist_th).ravel()

        if debug:
            print(f"{i:10d}, {ix_sub.sum():10.2f}, {consistency:10.2f}, {dist_th:10.2f}")

        # check the convergence after the first niter0 step
        if (i > niter0) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **kwargs)
        gp.optimize()
        consistency = alpha_ / chi2(p + 2).cdf(eta_sq1)
        mean, var = gp.predict(X)
        dist = (Y - mean)**2 / var

        iter_num += 1
        if callback is not None:
            callback(gp, consistency, iter_num, *callback_args)

    if debug:
        n0 = niter0 + 1
        n1 = iter_num - n0

    # reweighting step
    for i in range(niter1, niter1 + niter2):
        eta_sq2 = chi2(p).ppf(alpha2)
        ix_sub = (dist <= eta_sq2 * consistency).ravel()

        if debug:
            print(f"{i:10d}, {ix_sub.sum():10.2f}, {consistency:10.2f}")

        if (i > niter1) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **kwargs)
        gp.optimize()
        consistency = alpha2 / chi2(p + 2).cdf(eta_sq2)
        mean, var = gp.predict(X)
        dist = (Y - mean)**2 / var

        iter_num += 1
        if callback is not None:
            callback(gp, consistency, iter_num, *callback_args)

    if debug:
        n2 = iter_num - n1 - n0
        print(f'{n0+n1+n2}:\t{n0},\t{n1},\t{n2}')

    # outlier detection
    score = (dist / consistency)**0.5

    return gp, consistency, score


def ITGPv2(X, Y, alpha1=0.50, alpha2=0.95, nshrink=5, maxiter=20, reweight=1,
           callback=None, callback_args=(), optimize_kwargs={}, **kwargs):
    """
    Robust Gaussian Process Regression Based on Iterative Trimming.

    Parameters
    ----------
    X: array shape (n, p)
    Y: array shape (n, 1)
        Input data.
    alpha1, alpha2:
        Coverage fraction used in contraction step and reweighting step respectively.
    nshrink:
        Number of shrinking iteration.
    maxiter, reweight:
        Maximum iteration allowed in contraction step and reweighting step respectively.
    callback: callable
        Function for monitoring the iteration process. It takes
        the iteration number i and the locals() dict as input
        e.g.
            callback=lambda i, locals: print(i, locals['gp'].num_data, locals['gp'].param_array)
        or
            callback=lambda i, locals: locals['gp'].plot()
    callback_args:
        Extra parameters for callback.
    optimize_kwargs:
        GPy.core.GP.optimize parameters.
    **kwargs:
        GPy.core.GP parameters.

    Returns
    -------
    ITGPResult object, including gp, consistency, score, Y_avg, Y_var, ix_sub, iter_num
        gp:
            GPy.core.GP object.
        consistency:
            Consistency factor.
        score:
            Scaled residuals.
        Y_avg, Y_var:
            Expectation and variance of input data.
        ix_sub:
            Boolean index for trimming sample.
        iter_num:
            Total iterations performed.
    """
    n, p = Y.shape
    if p != 1:
        raise ValueError("Y is expected in shape (n, 1).")
    if n * alpha1 - 0.5 < 2:
        raise ValueError("The data set is unreasonably small!")
    if nshrink < 1:
        raise ValueError("nshrink >= 1 is expected.")
    if maxiter < nshrink:
        raise ValueError("maxiter >= nshrink is expected.")

    kwargs.setdefault('likelihood', GPy.likelihoods.Gaussian(variance=1.0))
    kwargs.setdefault('kernel', GPy.kern.RBF(X.shape[1]))
    kwargs.setdefault('name', 'ITGP regression')

    # initialization
    d_sq = None
    ix_old = None
    iter_num = 0

    # contraction step
    for i in range(maxiter):
        if i == 0:
            # starting with the full sample
            ix_sub = slice(None)
            consistency = 1.0
        else:
            # reducing alpha from 1 to alpha1 gradually
            if i < nshrink:
                alpha = alpha1 + (1 - alpha1) * (1 - i / nshrink)
            else:
                alpha = alpha1
            chi_sq = chi2(p).ppf(alpha)
            h = int(max(np.floor(n * alpha - 0.5), 0))  # (h+0.5)/n <= alpha
            # h = int(np.ceil(n * alpha - 0.5))  # (h+0.5)/n <= alpha

            ix_sub = (d_sq <= np.partition(d_sq, h)[h])  # alpha-quantile
            consistency = alpha / chi2(p + 2).cdf(chi_sq)

        # check the convergence after the first nshrink step
        if (i > nshrink) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **kwargs)
        gp.optimize(**optimize_kwargs)
        Y_avg, Y_var = gp.predict(X, include_likelihood=True)
        d_sq = ((Y - Y_avg)**2 / Y_var).ravel()

        if callback is not None:
            callback(iter_num, locals(), *callback_args)
            iter_num += 1

    # reweighting step
    for i in range(reweight):
        alpha = alpha2
        chi_sq = chi2(p).ppf(alpha)

        ix_sub = (d_sq <= chi_sq * consistency)
        consistency = alpha / chi2(p + 2).cdf(chi_sq)

        if (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **kwargs)
        gp.optimize(**optimize_kwargs)
        Y_avg, Y_var = gp.predict(X, include_likelihood=True)
        d_sq = ((Y - Y_avg)**2 / Y_var).ravel()

        if callback is not None:
            callback(iter_num, locals(), *callback_args)
            iter_num += 1

    # outlier detection
    score = (d_sq / consistency)**0.5

    return ITGPResult(gp, consistency, score, Y_avg, Y_var, ix_sub, iter_num)


ITGPResult = namedtuple('ITGPResult', ('gp', 'consistency', 'score', 'Y_avg', 'Y_var', 'ix_sub', 'iter_num'))


def ITGPv3(X, Y, alpha1=0.50, alpha2=0.975, nshrink=5, reweight=True,
           maxiter=None, predict=True, callback=None, callback_args=(),
           warm_start=True, optimize_kwargs={}, **gp_kwargs):
    """
    Robust Gaussian Process Regression Based on Iterative Trimming.

    Parameters
    ----------
    X: array shape (n, p)
    Y: array shape (n, 1)
        Input data with shape (# of data, # of dims).
    alpha1, alpha2: float in (0, 1)
        Coverage fraction used in shrinking and reweighting step respectively.
    nshrink, reweight: int (>=0)
        Number of shrinking and reweighting iterations respectively.
    maxiter: None, int (> nshrink)
        Maximum iteration allowed in shrinking step. If None (default), nshrink + 1 is used.
    predict: bool
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
        From which step, it uses the warm start for optimizing hyper parameters.
        Not implemented yet.
    optimize_kwargs:
        GPy.core.GP.optimize parameters.
    **gp_kwargs:
        GPy.core.GP parameters, including likelihood and kernel.
        Gaussian and RBF are used as defaults.

    Returns
    -------
    ITGPResult object, including gp, consistency, score, Y_avg, Y_var, ix_sub, iter_num.
        gp:
            GPy.core.GP object.
        consistency:
            Consistency factor.
        score:
            Scaled residuals.
        Y_avg, Y_var:
            Expectation and variance of input data.
        ix_sub:
            Boolean index for trimming sample.
        iter_num:
            Total iterations performed.
    """
    # checking
    n, p = Y.shape
    if p != 1:
        raise ValueError("Y is expected in shape (n, 1).")
    if n * alpha1 - 0.5 <= 2:
        raise ValueError("The dataset is unreasonably small.")
    if nshrink < 0 or reweight < 0:
        raise ValueError("nshrink >= 0 and reweight >= 0 are expected.")
    if maxiter is None or nshrink == 0:
        maxiter = nshrink + 1
        # make no sense to set maxiter > 1 when nshrink == 0
    elif maxiter <= nshrink:
        raise ValueError("maxiter > nshrink is expected.")
    if warm_start != 1:
        raise NotImplementedError("Not implemented yet.")

    gp_kwargs.setdefault('likelihood', GPy.likelihoods.Gaussian(variance=1.0))
    gp_kwargs.setdefault('kernel', GPy.kern.RBF(X.shape[1]))
    gp_kwargs.setdefault('name', 'ITGP regression')

    # temp vars declaration
    d_sq = None
    ix_old = None
    iter_num = 0

    # contraction step
    for i in range(maxiter):
        if i == 0:
            # starting with the full sample
            ix_sub = slice(None)
            consistency = 1.0
        else:
            # reducing alpha from 1 to alpha1 gradually
            if i < nshrink:
                alpha = alpha1 + (1 - alpha1) * (1 - i / nshrink)
            else:
                alpha = alpha1
            chi_sq = chi2(p).ppf(alpha)
            h = int(min(np.ceil(n * alpha - 0.5), n - 1))  # alpha <=(h+0.5)/n

            # XXX: might be buggy when there are identical data points
            # better to use argpartition! but may break ix_sub == ix_old.
            ix_sub = (d_sq <= np.partition(d_sq, h)[h])  # alpha-quantile
            consistency = alpha / chi2(p + 2).cdf(chi_sq)

        # check convergence when maxiter > nshrink + 1
        if (i > nshrink) and (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **gp_kwargs)
        gp.optimize(**optimize_kwargs)

        Y_avg, Y_var = gp.predict(X, include_likelihood=True)
        d_sq = ((Y - Y_avg)**2 / Y_var).ravel()

        if callback is not None:
            callback(iter_num, locals(), *callback_args)
            iter_num += 1

    # reweighting step
    for i in range(reweight):
        alpha = alpha2
        chi_sq = chi2(p).ppf(alpha)

        # XXX: might be buggy when there are identical data points
        ix_sub = (d_sq <= chi_sq * consistency)
        consistency = alpha / chi2(p + 2).cdf(chi_sq)

        if (ix_sub == ix_old).all():
            break  # converged
        ix_old = ix_sub

        gp = GPy.core.GP(X[ix_sub], Y[ix_sub], **gp_kwargs)
        gp.optimize(**optimize_kwargs)

        if i < reweight - 1 or predict:
            Y_avg, Y_var = gp.predict(X, include_likelihood=True)
            d_sq = ((Y - Y_avg)**2 / Y_var).ravel()
        else:
            pass  # skip final training unless prediction is wanted

        if callback is not None:
            callback(iter_num, locals(), *callback_args)
            iter_num += 1

    if predict:
        # outlier detection
        score = (d_sq / consistency)**0.5
        return ITGPResult(gp, consistency, score, Y_avg, Y_var, ix_sub, iter_num)
    else:
        return ITGPResult(gp, consistency, None, None, None, ix_sub, iter_num)


ITGPResult = namedtuple('ITGPResult', ('gp', 'consistency', 'score', 'Y_avg', 'Y_var', 'ix_sub', 'iter_num'))
