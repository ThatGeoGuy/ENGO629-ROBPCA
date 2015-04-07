#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Uses a robust pca measure by applying MCD to segment an egg from a table.
"""

import argparse
import sys

import numpy as np
from scipy.spatial import KDTree
from sklearn.covariance import MinCovDet

from engo629.classic_pca import principal_components

def main(filename, alpha, tol):
    """
    Main procedure. Runs a program to segment objects with dissimilar
    principal components from the robust PCA of the entire set of points.

    Parameters
    ----------

    filename : Name of file that holds some point cloud X, which will
               constitute our data matrix.

    Returns
    -------

    Zero if the program completes successfully.
    """
    if not (0.5 <= alpha <= 1.0):
        raise ValueError("segment-egg.py: The support fraction must be between 0.5 and 1")

    # Load data and insert it into a KDTree
    kdt = KDTree(np.loadtxt(filename))
    # Apply MCD method to full data
    mcd = MinCovDet(support_fraction=alpha).fit(kdt.data)

    # PCs of full (global) dataset
    gL, gPC = principal_components(None, lambda x: mcd.covariance_)

    print("Global planarity is:\n{}".format(gPC[:, 2]))

    marked_data        = np.zeros((kdt.data.shape[0], 5))
    makred_data[:,0:3] = kdt.data.copy()

    for i, pt in enumerate(kdt.data):
        dists, nn_indices = kdt.query(pt, k=8, p=2)

        LL, LPC = principal_components(kdt.data[nn_indices, :])

        # Calculate angle between planar components, then convert
        # to degrees
        theta = np.acos(np.dot(LPC[:, 2], gPC[:, 2]) /
                (np.norm(LPC[:, 2]) * np.norm(gPC[:, 2])))
        theta *= 180 / np.pi

        # Check that the normals aren't out of phase
        if theta >= 180:
            theta -= 180
        elif theta <= -180:
            theta += 180

        if theta > tol:
            marked_data[i, 3] = 0
        else:
            marked_data[i, 3] = 255

        marked_data[i, 4] = theta

    np.savetxt(sys.stdout, marked_data, '%.6f')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Performs object segmentation using Robust PCA')
    parser.add_argument('xyz_file',
            metavar='FILE.XYZ',
            type=str,
            help='File to perform segmentation over.')
    parser.add_argument('--alpha',
            type=float,
            default=0.75,
            help='Support fraction (alpha) of robust PCA')
    parser.add_argument('--tol',
            metavar='TOLERANCE',
            type=int,
            help='Tolerance between global and local plane angles in degrees')

    args = parser.parse_args()

    main(args.xyz_file, args.alpha, args.tol)
