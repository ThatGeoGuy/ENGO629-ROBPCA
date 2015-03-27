#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Defines a class which computes the ROBPCA method as defined by Mia Hubert,
Peter J. Rousseeuw and Karlien Vandem Branden (2005)
"""

import numpy as np

class ROBPCA(object):
    """
    Implements the ROBPCA algorithm as defined by Mia Hubert, Peter J.
    Rousseeuw, and Karlien Vandem Branden (2005)
    """

    def __init__(self, X, kmax=10, alpha=0.75):
        """
        Initializes the class instance with the data you wish to compute the
        ROBPCA algorithm over.

        Arguments:
        ----------

        X : An n x p data matrix (where n is number of data points and p is
            number of dimensions in data) which is to be reduced.

        kmax : Maximal number of components that will be computed. Set to 10
               by default

        alpha : Assists in determining step 2. The higher alpha is, the more
                efficient the estimates will be for uncontaminated data.
                However, lower values for alpha make the algorithm more robust.
                Can be any real value in the range [0.5, 1].
        """
        if not (0.5 <= alpha <= 1.0):
            raise ValueError("ROBPCA: alpha must be a value in the range [0.5, 1.0]")

        self.data  = X
        self.kmax  = kmax
        self.alpha = alpha
        return

    def reduce_to_affine_subspace(self):
        """
        Takes the mean-centred data-matrix and computes the affine subspace
        spanned by n observations. That is, we take the singular value
        decomposition of the mean-centred data-matrix and use Z = UD as our
        new data matrix where Z is an n by r0 sized matrix.

        Returns:
        --------

        Z : Z is the product of U and D of the singular value decomposition of
            the mean-centred data matrix
        """
        centred_data = self.data - np.mean(self.data, axis=0)
        U, D, V = np.linalg.svd(centred_data, False)
        return U * D

    def num_least_outlying_points(self):
        """
        Determines the least number of outlying points h, which should be less
        than n, our number of data points. `h` is computed as the maximum of either:

            alpha * n           OR
            (n + kmax + 1) / 2

        Returns:
        --------

        h : number of least outlying points.
        """
        n = self.data.shape[0]
        return np.max([
            self.alpha * n,
            (n + self.kmax + 1) / 2,
            ])
