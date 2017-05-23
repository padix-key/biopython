# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import Atom, AtomArray, AtomArrayStack
from . import vector_dot

def rmsd(reference: AtomArray, subject):
    sq_euclidian = _sq_euclidian(reference, subject)
    return np.sqrt(np.mean(sq_euclidian, axis=-1))
    
def rmsf(reference: AtomArray, subject: AtomArrayStack):
    if type(subject) != AtomArrayStack:
        raise ValueError("Subject must be AtomArrayStack")
    sq_euclidian = _sq_euclidian(reference, subject)
    return np.sqrt(np.mean(sq_euclidian, axis=0))

def average(atom_arrays: AtomArrayStack):
    mean_array = atom_arrays[0].copy()
    mean_array.pos = np.mean(atom_arrays.pos, axis=0)
    return mean_array
    
def _sq_euclidian(reference, subject):
    if type(reference) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    dif = subject.pos - reference.pos
    return vector_dot(dif, dif)