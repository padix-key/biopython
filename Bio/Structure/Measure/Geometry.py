# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from .. import *

def bond_length(atoms):
    dif = np.diff(atoms.pos, axis=0)
    product = dif * dif
    dist = np.sqrt(product[:,0] + product[:,1] + product[:,2])
    return dist

def get_centroid(atoms):
    struc_type = ensure_structure_type(atoms)
    if struc_type == "single":
        return atoms.pos
    if struc_type == "array":
        return np.mean(atoms.pos, axis=0)
    if struc_type == "stack":
        return np.mean(atoms.pos, axis=1)