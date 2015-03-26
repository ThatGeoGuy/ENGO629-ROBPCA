#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Tests functionality in robpca/pca.py
"""
import unittest
import numpy as np

from robpca import pca

class TestClassicalPCA(unittest.TestCase):
    """
    Tests function to compute classical principal components analysis.
    """

    def test_normal_to_z_plane(self):
        """
        Tests whether computing PCA over a point cloud distributed in the plane
        Z = 0 provides the unit vector [0, 0, 1].
        """

        # Initialize data matrix
        X       = np.zeros((1000, 3))
        X[:, 0] = np.random.rand(1000)
        X[:, 1] = np.random.rand(1000)

        # Calculate PCs and respective variances
        variances, PCs = pca.principal_components(X)

        min_var        = np.min(variances)
        pc_min_var     = PCs[:, np.argmin(variances)]

        # Expected outputs
        expect_min_var = 0
        expect_pc      = np.array([0,0,1])

        self.assertEqual(min_var, expect_min_var)
        self.assertTrue(np.all(pc_min_var == expect_pc))
        return
