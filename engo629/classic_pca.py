#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Defines functions that deal with classical (non-robust) principal components
analysis.
"""

import numpy as np

def principal_components(X, cov_func=None):
    """
    Calculates the principal components of data stored in an array X of size
    n by p, where n (the number of rows) corresponds to the number of data
    fields, and p corresponds to the number of dimensions.

    Args:
    -----

    X        : an n by p array-like object. 'p' number of principal components and
               their variances will be returned.
    cov_func : A function which can be passed in to calculate the covariance
               matrix of the data in X. It should just accept X, and defaults to
               Numpy's covariance function `numpy.cov`. Note that np.cov accepts
               transpose(X) by default, but this interface assumes the opposite.

    Return:
    -------

    L   : Variance component of principal component in descending order
    PCs : Principal component vectors. Each column corresponding to the
          respective variance component is the unit vector of the principal
          component. Sorted and returned in descending order of L.

    """
    if not cov_func:
        cov_func = lambda x: np.cov(x.T)

    L, PCs    = np.linalg.eigh(cov_func(X))
    arg_order = np.argsort(L)[::-1]
    return L[arg_order], PCs[:, arg_order]
