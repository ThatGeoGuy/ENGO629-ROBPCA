#!/usr/bin/env python3

"""
Defines functions that deal with classical (non-robust) principal components
analysis.
"""

import numpy as np

def principal_components(X, cov_func=np.cov):
    """
    Calculates the principal components of data stored in an array X of size
    n by p, where n (the number of rows) corresponds to the number of data
    fields, and p corresponds to the number of dimensions.

    Args:
    -----

    X : an n by p array-like object. 'p' number of principal components and
        their variances will be returned.

    cov_func : A function which can be passed in to calculate the covariance
               matrix of the data in X. It should accept the transpose of X,
               and defaults to Numpy's covariance function `numpy.cov`

    Return:
    -------

    evals : Variance component of principal component

    evecs : Principal component vectors. Each column corresponding to the
            respective variance component is the unit vector of the principal
            component.

    """
    evals, evecs = np.linalg.eigh(cov_func(X.T))
    return evals, evecs
