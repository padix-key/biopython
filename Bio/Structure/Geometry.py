# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import Atom, AtomArray, AtomArrayStack

def distance(atoms1, atoms2):
    dif = atoms2.pos - atoms1.pos
    product = dif * dif
    dist = np.sqrt(product[...,0] + product[...,1] + product[...,2])
    return dist

def centroid(atoms):
    if type(atoms) == Atom:
        return atoms.pos
    if type(atoms) == AtomArray:
        return np.mean(atoms.pos, axis=0)
    if type(atoms) == AtomArrayStack:
        return np.mean(atoms.pos, axis=1)
    
