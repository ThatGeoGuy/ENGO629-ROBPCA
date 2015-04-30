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
import matplotlib.pyplot as plt
from numpy import reshape
from scipy import ndimage
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors

from engo629.classic_pca import principal_components

def bounding_box(data):
    """
    Cuts a bounding box of data out of data and returns the upper left and
    lower right coordinates.
    """
    plt.imshow(reshape(data, (424, 512, 3)))
    (x1, y1), (x2, y2) = plt.ginput(2)
    plt.close(1)
    return int(x1), int(y1), int(x2), int(y2)

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

    x1, y1, x2, y2 = bounding_box(data)
    data_bb = reshape(reshape(data, (424, 512, 3))[y1:y2, x1:x2, :], (-1, 3)).copy()

    kNN  = NearestNeighbors(n_neighbors = 8, p = 2).fit(data_bb)

    if classic:
        loc     = np.mean(data_bb, axis= 0)
        gL, gPC = principal_components(data_bb)
    else:
        # Apply MCD method to full data
        mcd = MinCovDet(support_fraction=alpha).fit(data_bb)
        loc = mcd.location_

        # PCs of full (global) dataset
        gL, gPC = principal_components(None, lambda x: mcd.covariance_)

    proj = abs(np.dot(loc, gPC[:,2]))

    # Verbosity. This isn't really needed, and I am not yet going to add a flag
    # for this
    # print("Global planarity is:\n{}".format(gPC[:, 2]))
    # print("Global eigenvalues: {}".format(gL))
    # print("Location is: {}".format(loc))

    marked_data = 255 * np.ones((data_bb.shape[0], 4))
    marked_data[:,:3] = data_bb.copy()

    for i, pt in enumerate(data_bb):
        dists, nn_indices = kNN.kneighbors(pt, n_neighbors=8)

        LL, LPC = principal_components(data_bb[nn_indices[0], :])

        # Calculate angle between planar components, then convert
        # to degrees
        # A <dot> B = |A| * |B| * cos(theta)
        theta = np.arccos(np.dot(LPC[:, 2], gPC[:, 2]) /
                (np.linalg.norm(LPC[:, 2]) * np.linalg.norm(gPC[:, 2])))
        theta *= 180 / np.pi

        loc  = abs(np.dot(pt, gPC[:,2]))
        dist = loc - proj

        # Check that the normals aren't out of phase
        if theta > 90:
            theta -= 180
        if theta < -90:
            theta += 180

        if ((abs(theta) > ttol) and (LL[2] > gL[2])) or dist < -dtol:
            marked_data[i,3] = 0
        if dist > dtol or dist > 0:
            marked_data[i,3] = 255

    img = reshape(marked_data[:,3], (y2-y1, x2-x1))
    img = ndimage.binary_dilation(img)
    marked_data[:,3] = reshape(img, -1)
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
