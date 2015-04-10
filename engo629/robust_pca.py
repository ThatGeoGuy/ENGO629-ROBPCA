#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Defines a class which computes the ROBPCA method as defined by Mia Hubert,
Peter J. Rousseeuw and Karlien Vandem Branden (2005)
"""

import sys
import numpy as np
from sklearn.covariance import fast_mcd, MinCovDet
from np.random import choice

from .classic_pca import principal_components

class ROBPCA(object):
    """
    Implements the ROBPCA algorithm as defined by Mia Hubert, Peter J.
    Rousseeuw, and Karlien Vandem Branden (2005)
    """

    def __init__(self, X, k=0, kmax=10, alpha=0.75, mcd=True):
        """
        Initializes the class instance with the data you wish to compute the
        ROBPCA algorithm over.

        Parameters
        ----------

        X      : An n x p data matrix (where n is number of data points and p
                 is number of dimensions in data) which is to be reduced.
        k      : Number of principal components to compute. If k is missing,
                 the algorithm itself will determine the number of components
                 by finding a k such that L_k / L_1 > 1e-3 and (sum_1:k L_j /
                 sum_1:r L_j >= 0.8)
        kmax   : Maximal number of components that will be computed. Set to 10
                 by default
        alpha  : Assists in determining step 2. The higher alpha is, the more
                 efficient the estimates will be for uncontaminated data.
                 However, lower values for alpha make the algorithm more
                 robust. Can be any real value in the range [0.5, 1].
        mcd    : Specifies whether or not to use the MCD covariance matrix to
                 compute the principal components when p << n.
        """
        if k < 0:
            raise ValueError("ROBPCA: number of dimensions k must be greater than or equal to 0.")
        if kmax < 1:
            raise ValueError("ROBPCA: kmax must be greater than 1 (default is 10).")
        if not (0.5 <= alpha <= 1.0):
            raise ValueError("ROBPCA: alpha must be a value in the range [0.5, 1.0].")
        if mcd is not True or mcd is not False:
            raise ValueError("ROBPCA: mcd must be either True or False.")

        if k > kmax:
            print("ROBPCA: WARNING - k is greater than kmax, setting k to kmax.",
                    file=sys.stderr)
            k = kmax

        self.data   = X
        self.k      = k
        self.kmax   = kmax
        self.alpha  = alpha
        self.mcd    = mcd
        return

    @staticmethod
    def reduce_to_affine_subspace(X):
        """
        Takes the data-matrix and computes the affine subspace spanned by n
        observations of the mean-centred data.

        Parameters
        ----------

        X : X is an n by p data matrix where n is the number of observations
            and p is the number of dimensions in the data.

        Returns
        --------

        Z   : Z is the affine subspace of the data matrix X. It is the same
              data as X but represents itself within its own dimensionality.
        rot : Specifies the PCs computed here that were used to rotate X into
              the subspace Z.
        """
        # Compute regular PCA
        # L  -> lambdas (eigenvalues)
        # PC -> principal components (eigenvectors)
        _, PC  = principal_components(X)
        centre = np.mean(X, axis=0)

        # New data matrix
        Z = np.dot((X - centre), PC)

        return Z, PC

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

    def find_least_outlying_points(self, X):
        """
        Finds the `h` number of points in the dataset that are least-outlying.
        Does this by first computing the modified Stahel-Donoho
        affine-invariant outlyingness.

        Parameters
        ----------

        X : The data matrix with which you want to find the least outlying
            points using the modified Stahel-Donoho outlyingness measure

        Returns
        --------

        H0 : indices of data points from X which have the least
             outlyingness
        """
        n, p           = X.shape
        self.h         = num_least_outlying_points()
        num_directions = min(250, n * (n - 1) / 2)

        B = np.array([ROBPCA.direction_through_hyperplane(X)
                        for _ in range(num_directions)])
        B_norm     = np.linalg.norm(B, axis = 1)
        index_norm = B_norm > 1e-12
        A          = np.dot(np.diag(1 / B_norm[index_norm]), B[index_norm, :])

        # Used as a matrix because there's a bug in fast_mcd / MinCovDet
        # The bug is basically due to single row / column arrays not
        # maintaining the exact shape information (e.g. np.zeros(3).shape
        # returns (3,) and not (3,1) or (1,3).
        Y = np.matrix(np.dot(Z, A.T))
        ny, ry = Y.shape

        # Set up lists for univariate t_mcd and s_mcd
        t_mcd = np.zeros(ry)
        s_mcd = np.zeros(ry)

        for i in range(ry):
            mcd = MinCovDet(support_fraction=self.alpha).fit(Y[:,i])
            t_mcd[i] = mcd.location_
            s_mcd[i] = mcd.covariance_

        # Supposedly if any of the s_mcd values is zero we're supposed to
        # project all the data points onto the hyperplane defined by the
        # direction orthogonal to A[i, :]. However, the reference
        # implementation in LIBRA does not explicitly do this, and quite
        # frankly the code is so terrible I've moved on for the time being.
        outl = np.max(np.abs(np.array(Y) - t_mcd) / s_mcd, axis=1)
        H0 = np.argsort(outl)[::-1][0:h]
        return H0

    def compute_pc(self):
        """
        Robustly computes the principal components of the data matrix.
        This is primarily broken up into one of several ways, depending on the
        dimensionality of the data (whether p > n or p < n)
        """
        X, rot = ROBPCA.reduce_to_affine_subspace(self.data)
        centre = np.mean(self.data, axis=0)

        if np.linalg.rank(X) == 0:
            raise ValueError("All data points collapse!")

        n, p   = X.shape
        self.h = num_least_outlying_points()

        # Use MCD instead of ROBPCA if p << n
        if p < min(np.floor(n / 5), self.kmax) and self.mcd:
            mcd = MinCovDet(support_fraction=self.alpha).fit(X)

            loc = mcd.location_
            cov = mcd.covariance_
            L, PCs = principal_components(X, lambda x: cov)
            result = {
                    'location'    : np.dot(rot,loc) + centre,
                    'covariance'  : np.dot(rot, cov),
                    'eigenvalues' : L,
                    'loadings'    : np.dot(rot, PCs),
                    }
            return result

        # Otherwise just continue with ROBPCA
        H0 = find_least_outlying_points(X)
        Xh = X[H0, :]

        Lh, Ph    = principal_components(Xh)
        centre_Xh = np.mean(Xh, axis=0)

        self.kmax = np.min(np.sum(Lh > 1e-12), self.kmax)

        # If k was not set or chosen to be 0, then we should calculate it
        # Basically we test if the ratio of the k-th eigenvalue to the 1st
        # eigenvalue (sorted decreasingly) is larger than 1e-3 and if the
        # fraction of cumulative dispersion is greater than 80%
        if self.k == 0:
            test, = np.where(Lh / Lh[0] <= 1e-3)

            if len(test):
                self.k = min(np.sum(Lh > 1e-12), int(test + 1), self.kmax)
            else:
                self.k = min(np.sum(Lh > 1e-12), self.kmax)

            cumulative = np.cumsum(Lh[1:self.k]) / np.sum(Lh)
            if cumulative[self.k-1] > 0.8:
                self.k = int(np.where(cumulative >= 0.8)[0])

        centre += np.mean(Xh, axis=0)
        rot     = np.dot(rot, Ph)

        X2  = np.dot(X - np.mean(Xh, axis = 0), Ph)
        X2  = X2[:, 1:self.k]
        rot = rot[:, 1:self.k]

        mcd = MinCovDet(support_fraction=(self.h / n)).fit(X2)
        loc = mcd.location_
        cov = mcd.covariance_

        L, PCs = principal_components(X2, lambda x: cov)
        result = {
                'location'    : np.dot(rot, loc) + centre,
                'covariance'  : np.dot(rot,cov),
                'eigenvalues' : L,
                'loadings'    : np.dot(rot, PCs),
                }
        return result
