# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module allows checking of atom arrays and atom array stacks for errors in the structure.
"""

import numpy as np
from . import Atom, AtomArray, AtomArrayStack


def check_continuity(array):
    disc_i = []
    ids = array.res_id
    diff = np.diff(ids)
    discontinuity = np.where( ((diff != 0) & (diff != 1)) )
    return discontinuity[0]