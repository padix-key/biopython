# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np

def vector_dot(v1,v2):
    return (v1*v2).sum(axis=-1)

def norm_vector(v):
    factor = np.linalg.norm(v, axis=-1)
    if isinstance(factor, np.ndarray):
        v /= factor[..., np.newaxis]
    else:
        v /= factor