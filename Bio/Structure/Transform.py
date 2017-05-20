# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import get_centroid

def translate(atoms, vector):
        if len(vector) != 3:
            raise ValueError("Translation vector must be container of length 3")
        transformed = atoms.copy()
        transformed.pos += np.array(vector)
        return transformed

def rotate(atoms, angles):
    from numpy import sin, cos
    
    if len(angles) != 3:
        raise ValueError("Translation vector must be container of length 3")
    
    rot_x = np.array([[ 1,               0,               0               ],
                      [ 0,               cos(angles[0]),  -sin(angles[0]) ],
                      [ 0,               sin(angles[0]),  cos(angles[0])  ]])
    
    rot_y = np.array([[ cos(angles[1]),  0,               sin(angles[1])  ],
                      [ 0,               1,               0               ],
                      [ -sin(angles[1]), 0,               cos(angles[1])  ]])
    
    rot_z = np.array([[ cos(angles[2]),  -sin(angles[2]), 0               ],
                      [ sin(angles[2]),  cos(angles[2]),  0               ],
                      [ 0,               0,               1               ]])
    
    transformed = atoms.copy()
    transformed.pos = np.dot(transformed.pos, rot_x)
    transformed.pos = np.dot(transformed.pos, rot_y)
    transformed.pos = np.dot(transformed.pos, rot_z)
    return transformed

def rotate_centered(atoms, angles):
    transformed = atoms.copy()
    centroid = get_centroid(transformed)
    transformed.pos -= centroid
    transformed = rotate(transformed, angles)
    transformed.pos += centroid
    return transformed