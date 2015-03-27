#!/usr/bin/env python3
# This file is a part of ENGO629-ROBPCA
# Copyright (c) 2015 Jeremy Steward
# License: http://www.gnu.org/licenses/gpl-3.0-standalone.html GPL v3+

"""
Tests functionality in robpca/ROBPCA.py
"""

import unittest

from engo629.robust_pca import ROBPCA

class TestROBPCA(unittest.TestCase):
    """
    Tests the ROBPCA class and it's members. Should overall serve as tests to
    verify that the ROBPCA algorithm is working correctly and as intended.
    """
