# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import centroid

def translate(atoms, vector):
        if len(vector) != 3:
            raise ValueError("Translation vector must be container of length 3")
        transformed = atoms.copy()
        transformed.pos += np.array(vector)
        return transformed

def rotate(atoms, angles):
    from numpy import sin, cos
    # Check if "angles" contains 3 angles for all dimensions
    if len(angles) != 3:
        raise ValueError("Translation vector must be container of length 3")
    # Create rotation matrices for all 3 dimensions
    rot_x = np.array([[ 1,               0,               0               ],
                      [ 0,               cos(angles[0]),  -sin(angles[0]) ],
                      [ 0,               sin(angles[0]),  cos(angles[0])  ]])
    
    rot_y = np.array([[ cos(angles[1]),  0,               sin(angles[1])  ],
                      [ 0,               1,               0               ],
                      [ -sin(angles[1]), 0,               cos(angles[1])  ]])
    
    rot_z = np.array([[ cos(angles[2]),  -sin(angles[2]), 0               ],
                      [ sin(angles[2]),  cos(angles[2]),  0               ],
                      [ 0,               0,               1               ]])
    # Copy AtomArray(Stack) and apply rotations
    # Note that the coordinates are treated as row vector
    transformed = atoms.copy()
    transformed.pos = np.dot(transformed.pos, rot_x)
    transformed.pos = np.dot(transformed.pos, rot_y)
    transformed.pos = np.dot(transformed.pos, rot_z)
    return transformed

def rotate_centered(atoms, angles):
    # Rotation around centroid requires translation of centroid to origin
    transformed = atoms.copy()
    centro = centroid(transformed)
    transformed.pos -= centro
    transformed = rotate(transformed, angles)
    transformed.pos += centro
    return transformed