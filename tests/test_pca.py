#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Tests functionality in robpca/pca.py
"""
import unittest
from random import randint

import numpy as np

from robpca import pca

class TestClassicalPCA(unittest.TestCase):
    """
    Tests function to compute classical principal components analysis.
    """

    def test_normal_to_cardinal_plane(self):
        """
        Tests whether computing PCA over a set of data with one dimension set
        to constant yields a principal component of unit length with all
        dimension values set to zero except that of the dimension which is
        constant in the data matrix.

        e.g. if we have an XYZ point cloud and all Z values are set to zero,
        there is no variance in the Z direction therefore the PC corresponding
        to the plane defined by the point cloud should be [0,0,1]
        """

        # Data sizes and axis number (the axis that will be set to zero)
        #
        n    = randint(4, 1500)
        p    = randint(2, 100)
        axis = randint(0, p - 1)

        # Initialize data matrix
        X          = np.random.rand(n, p)
        X[:, axis] = 0

        # Calculate PCs and respective variances
        variances, PCs = pca.principal_components(X)
        min_var        = np.min(variances)
        pc_min_var     = PCs[:, np.argmin(variances)]

        # Expected outputs
        expect_min_var  = 0
        expect_pc       = np.zeros(p)
        expect_pc[axis] = 1

        # Machine precision and Tests
        # When testing the actual principal component, note that it is
        # necessary to test both the PC and it's complement (-1 * PC) because
        # the direction of a unit vector is arbitrary (e.g. Z axis can point up
        # or down, but it still defines the same axis)
        eps = abs(np.spacing(1))

        self.assertTrue(abs(min_var - expect_min_var) < eps)
        self.assertTrue(
            np.all(np.abs( pc_min_var - expect_pc) < 10 * eps) or
            np.all(np.abs(-pc_min_var - expect_pc) < 10 * eps)
        )
        return
