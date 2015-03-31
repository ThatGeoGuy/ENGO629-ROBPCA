#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Defines a class which computes the ROBPCA method as defined by Mia Hubert,
Peter J. Rousseeuw and Karlien Vandem Branden (2005)
"""

import numpy as np
from sklearn.covariance import fast_mcd
from np.random import choice

class ROBPCA(object):
    """
    Implements the ROBPCA algorithm as defined by Mia Hubert, Peter J.
    Rousseeuw, and Karlien Vandem Branden (2005)
    """

    def __init__(self, X, kmax=10, alpha=0.75, mcd=True):
        """
        Initializes the class instance with the data you wish to compute the
        ROBPCA algorithm over.

        Parameters
        ----------

        X      : An n x p data matrix (where n is number of data points and p
                 is number of dimensions in data) which is to be reduced.
        kmax   : Maximal number of components that will be computed. Set to 10
                 by default
        alpha  : Assists in determining step 2. The higher alpha is, the more
                 efficient the estimates will be for uncontaminated data.
                 However, lower values for alpha make the algorithm more robust.
                 Can be any real value in the range [0.5, 1].
        """
        if kmax < 1:
            raise ValueError("ROBPCA: kmax must be greater than 1 (default is 10).")
        if not (0.5 <= alpha <= 1.0):
            raise ValueError("ROBPCA: alpha must be a value in the range [0.5, 1.0]")

        self.data   = X
        self.kmax   = kmax
        self.alpha  = alpha
        return

    def reduce_to_affine_subspace(self):
        """
        Takes the mean-centred data-matrix and computes the affine subspace
        spanned by n observations.

        Returns
        --------

        Z : Z is the product of U and D of the singular value decomposition of
            the mean-centred data matrix
        """
        L, PC = np.linalg.eigh(np.cov(self.data.T))
        # Compute regular PCA
        L, PC  = np.linalg.eigh(np.cov(self.data.T))
        centre = np.mean(self.data, axis=0)

        # We want PCs from largest to smallest
        arg_order = np.argsort(L)[::-1]
        # New data matrix
        Z = np.dot((self.data - centre), PC[:,arg_order])

        return Z

    def num_least_outlying_points(self):
        """
        Determines the least number of outlying points h, which should be less
        than n, our number of data points. `h` is computed as the maximum of
        either:

            alpha * n           OR
            (n + kmax + 1) / 2

        Returns
        --------

        h : number of least outlying points.
        """
        n = self.data.shape[0]
        return int(np.max([self.alpha * n, (n + self.kmax + 1) / 2]))

    @staticmethod
    def direction_through_hyperplane(X):
        """
        Calculates a direction vector between two points in Z, where Z is an
        n x p matrix. This direction is projected upon to find the number of
        least outlying points using the Stahel-Donoho outlyingness measure.

        Parameters
        ----------

        X : Affine subspace of mean-centred data-matrix

        Returns
        --------

        p0 : point of origin of the direction vector d
        d : direction vector between two points in Z
        """
        n = Z.shape[0]
        p = Z.shape[1]

        d = None
        if n > p:
            P    = np.array(X[choice(n,p), :])
            Q, R = np.linalg.qr(P)

            if np.linalg.matrix_rank(Q) == p:
                d = np.linalg.solve(Q, np.ones(p))
        else:
            P   = np.array(X[choice(n,2), :])
            tmp = P[1, :] - P[0, :]
            N   = np.sqrt(np.dot(E,E))

            if N > 1e-8:
                d = tmp / N
        return d

    def compute_pc(self):
        """
        Robustly computes the principal components of the data matrix.
        This is primarily broken up into one of several ways, depending on the
        dimensionality of the data (whether p > n or p < n)
        """
        Z              = reduce_to_affine_subspace()
        n, p           = Z.shape
        self.h         = num_least_outlying_points()
        num_directions = min(250, n * (n - 1) / 2)

        # Find least outlying points --> Break this off later into separate function
        B = np.array([ROBPCA.direction_through_hyperplane(Z)
                        for _ in range(num_directions)])
        B_norm     = np.linalg.norm(B, axis = 1)
        index_norm = B_norm > 1e-12
        A          = np.dot(np.diag(1 / B_norm[index_norm]), B[index_norm, :])

        #
        Y = np.dot(Z, A.T)

        t_mcd = np.zeros()
        for _ in range(Y.shape[1]):



    def find_least_outlying_points(self, Z):
        """
        Finds the `h` number of points in the dataset that are least-outlying.
        Does this by first computing the modified Stahel-Donoho
        affine-invariant outlyingness.

        Parameters
        ----------

        Z : Affine subspace of mean-centred data-matrix.

        Returns
        --------

        H0 : set of data points from self.data which have the least
             outlyingness
        """
        loc = np.zeros(self.num_directions)
        cov = np.zeros(self.num_directions)
        v   = np.zeros((self.num_directions, Z.shape[1]))

        h = num_least_outlying_points()

        for i in range(self.num_directions):
            C       = np.matrix(np.dot(Z, v[i, :])).T
            v[i, :] = ROBPCA.direction_through_hyperplane(Z)
            loc[i], cov[i], _, _ = fast_mcd(C)

        outl = np.zeros(Z.shape[1])

