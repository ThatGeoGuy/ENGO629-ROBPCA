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
# from scipy.spatial import KDTree
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors

from engo629.classic_pca import principal_components

def main(filename, alpha, ttol, dtol, classic, ofile):
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
    data = np.loadtxt(filename)
    kNN  = NearestNeighbors(n_neighbors = 8, p = 2).fit(data)

    if classic:
        loc     = np.mean(data, axis= 0)
        gL, gPC = principal_components(data)
    else:
        # Apply MCD method to full data
        mcd = MinCovDet(support_fraction=alpha).fit(data)
        loc = mcd.location_

        # PCs of full (global) dataset
        gL, gPC = principal_components(None, lambda x: mcd.covariance_)

    proj = abs(np.dot(loc, gPC[:,2]))

    print("Global planarity is:\n{}".format(gPC[:, 2]))
    print("Global eigenvalues: {}".format(gL))
    print("Location is: {}".format(loc))

    marked_data        = np.ones((data.shape[0], 4))
    marked_data[:,0:3] = data.copy()

    for i, pt in enumerate(data):
        dists, nn_indices = kNN.kneighbors(pt, n_neighbors=8)

        LL, LPC = principal_components(data[nn_indices[0], :])

        # Calculate angle between planar components, then convert
        # to degrees
        # A <dot> B = |A| * |B| * cos(theta)
        theta = np.arccos(np.dot(LPC[:, 2], gPC[:, 2]) /
                (np.linalg.norm(LPC[:, 2]) * np.linalg.norm(gPC[:, 2])))
        theta *= 180 / np.pi

        loc  = abs(np.dot(pt, gPC[:,2]))
        dist = abs(loc - proj)

        # Check that the normals aren't out of phase
        if theta > 90:
            theta -= 180
        if theta < -90:
            theta += 180

        if ((abs(theta) > ttol) and (LL[2] > gL[2])) or dist > dtol:
            marked_data[i, 3] = 0

    np.savetxt(ofile, marked_data, '%.6f')
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
    parser.add_argument('--ttol',
            metavar='THETA TOL',
            type=int,
            default=5,
            help='Tolerance between global and local plane angles in degrees')
    parser.add_argument('--dtol',
            metavar='DIST TOL',
            type=float,
            default=1e-2,
            help='Location tolerance between point and background plane')
    parser.add_argument('--classic',
            action='store_true',
            default=False,
            help='Tells the algorithm to use robust PCA or classic (robust is default)')
    parser.add_argument('-o',
            metavar='OUTPUT',
            type=str,
            default='output.txt',
            help='Name of the file you want to output the data to')

    args = parser.parse_args()

    main(args.xyz_file, args.alpha, args.ttol, args.dtol, args.classic, args.o)
