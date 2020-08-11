import GPy
import numpy as np
from scipy.stats import chi2

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

    if callback is not None:
        callback(gp, consistency, 0, *callback_args)

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

        if callback is not None:
            callback(gp, consistency, i + 1, *callback_args)

    n0 = niter0
    n1 = i + 1 - niter0 if niter1 else 0

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

        if callback is not None:
            callback(gp, consistency, i + 1, *callback_args)

    n2 = i + 1 - niter1
    print(f'{n0+n1+n2}:\t{n0},\t{n1},\t{n2}')

    # outlier detection
    score = (dist / consistency)**0.5


    return gp, consistency, score
