# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import Atom, AtomArray, AtomArrayStack
from . import vector_dot

def rmsd(reference, subject):
    if type(reference) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    
    dif = subject.pos - reference.pos
    sq_euclidian = vector_dot(dif, dif)
    if type(subject) == AtomArray:
        return np.sqrt(np.mean(sq_euclidian))
    elif type(subject) == AtomArrayStack:
        # TODO
        pass
    else:
        raise ValueError("Subject must be AtomArray or AtomArrayStack")