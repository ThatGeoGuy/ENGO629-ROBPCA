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

    def __init__(self, X):
        """
        Initializes the class instance with the data you wish to compute the
        ROBPCA algorithm over.
        """
        self.data = X

    @staticmethod
    def reduce_to_affine_subspace(data):
        """
        Takes the mean-centred data-matrix and computes the affine subspace
        spanned by n observations. That is, we take the singular value
        decomposition of the mean-centred data-matrix and use Z = UD as our
        new data matrix where Z is an n by r0 sized matrix.

        Arguments:
        ----------

        data : An n x p data matrix (where n is number of data points and p is
               number of dimensions in data) which is to be reduced.

        Returns:
        --------

        Z : Z is the product of U and D of the singular value decomposition of
            the mean-centred data matrix
        """
        centred_data = data - np.mean(self.data, axis=0)
        U, D, V = np.linalg.svd(centred_data, False)
        return U * D
