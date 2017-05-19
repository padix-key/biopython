# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from .. import *

def rmsd(reference, subject):
    if not isinstance(reference, AtomArray):
        raise ValueError("Reference must be AtomArray")
    struc_type = ensure_structure_type(subject, allow_single=False)
    
    dif = subject.pos - reference.pos
    product = dif * dif
    if struc_type == "array":
        sq_euclidian = product[:,0] + product[:,1] + product[:,2]
        return np.sqrt(np.mean(sq_euclidian))
    else:
        sq_euclidian = product[:,:,0] + product[:,:,1] + product[:,:,2]
        # TODO