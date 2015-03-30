#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Defines a class which computes the ROBPCA method as defined by Mia Hubert,
Peter J. Rousseeuw and Karlien Vandem Branden (2005)
"""

import numpy as np
from sklearn.covariance import MinCovDet
from np.random import choice

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
        U, s, V = np.linalg.svd(centred_data, False)
        S = np.diag(s)
        return U * S

    def num_least_outlying_points(self):
        """
        Determines the least number of outlying points h, which should be less
        than n, our number of data points. `h` is computed as the maximum of
        either:

            alpha * n           OR
            (n + kmax + 1) / 2

        Returns:
        --------

        h : number of least outlying points.
        """
        n = self.data.shape[0]
        return np.max([self.alpha * n, (n + self.kmax + 1) / 2])

    @staticmethod
    def direction_coefficients_through_hyperplane(Z):
        """
        Calculates a direction vector between two points in Z, where Z is an
        n x p matrix. This direction is projected upon to find the number of
        least outlying points using the Stahel-Donoho outlyingness measure.

        Arguments:
        ----------

        Z : Affine subspace of mean-centred data-matrix

        Returns:
        --------

        p0 : point of origin of the direction vector d
        d : direction vector between two points in Z
        """
        n = Z.shape[0]
        p = Z.shape[1]

        d = None
        if n > p:
            P    = np.array(Z[choice(n,p), :])
            Q, R = np.linalg.qr(P)

            if np.linalg.matrix_rank(Q) == p:
                d = np.linalg.solve(Q, np.ones(p))
        else:
            P   = np.array(Z[choice(n,2), :])
            tmp = P[1, :] - P[0, :]
            N   = np.sqrt(np.dot(E,E))

            if N > 1e-8:
                d = tmp / N
        return d

    def find_least_outlying_points(self, Z):
        """
        Finds the `h` number of points in the dataset that are least-outlying.
        Does this by first computing the modified Stahel-Donoho
        affine-invariant outlyingness.

        Arguments:
        ----------

        Z : Affine subspace of mean-centred data-matrix.

        Returns:
        --------

        H0 : set of data points from self.data which have the least
             outlyingness
        """
