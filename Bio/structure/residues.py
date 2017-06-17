# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module provides utility for handling data on residue level, rather than
atom level.
"""

import numpy as np
from . import AtomArray, AtomArrayStack

def apply_residue_wise(array, data, function, shape=None, axis=None):
    ids = array.res_id
    # Maximum length of the processed data is length of atom array
    if shape == None:
        processed_data = np.zeros(len(ids))
    else:
        processed_data = np.zeros((len(ids),) + shape)
    start = 0
    i = 0
    while start in range(len(ids)):
        stop = start
        # When first condition fails the second one is not checked,
        # otherwise the second condition could throw IndexError
        while stop < len(ids) and ids[stop] == ids[start]:
            stop += 1
        interval = data[start:stop]
        if axis == None:
            value = function(interval)
        else:
            value = function(interval, axis=axis)
        processed_data[i] = value
        start = stop
        i += 1
    # Trim processed_data to correct size
    return processed_data[:i]

def get_residues(array):
    # Maximum length is length of atom array
    ids = np.zeros(len(array.res_id), dtype=float)
    names = np.zeros(len(array.res_id), dtype="U3")
    ids[0] = array.res_id[0]
    names[0] = array.res_name[0]
    i = 1
    for j in range(len(array.res_id)):
        if array.res_id[j] != ids[i-1]:
            ids[i] = array.res_id[j]
            names[i] = array.res_name[j]
            i += 1
    # Trim to correct size
    return ids[:i], names[:i]
            