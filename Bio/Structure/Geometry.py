# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import Atom, AtomArray, AtomArrayStack
from . import position
from . import vector_dot, norm_vector
from . import BadStructureError

def distance(atoms1, atoms2):
    v1 = position(atoms1)
    v2 = position(atoms2)
    
    if len(v1.shape) <= len(v2.shape):
        dif = v2 - v1
    else:
        dif = v1 - v2
    dist = np.sqrt(vector_dot(dif, dif))
    return dist


def centroid(atoms):
    return np.mean(position(atoms), axis=-2)


def angle(atom1, atom2, atom3):
    v1 = position(atom2) - position(atom1)
    v2 = position(atom3) - position(atom2)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1,v2))


def dihedral(atom1, atom2, atom3, atom4):
    v1 = position(atom2) - position(atom1)
    v2 = position(atom3) - position(atom2)
    v3 = position(atom4) - position(atom3)
    norm_vector(v1)
    norm_vector(v2)
    norm_vector(v3)
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    
    return np.arctan2(y,x)


def dihedral_backbone(atom_array, chain_id):
    try:
        backbone = atom_array[((atom_array.atom_name == "N") |
                               (atom_array.atom_name == "CA") |
                               (atom_array.atom_name == "C")) &
                               (atom_array.hetero == " ") &
                               (atom_array.chain_id == chain_id)]
        angle_atoms = np.zeros(( (backbone.seq_length(chain_id)-1)*3, 4, 3 ))
        for i in range(len(angle_atoms)):
            angle_atoms[i] = backbone.pos[0+i : 4+i]
        dihed = dihedral(angle_atoms[:,0], angle_atoms[:,1],
                         angle_atoms[:,2], angle_atoms[:,3])
        psi = dihed[0::3]
        omega = dihed[1::3]
        phi = dihed[2::3]
        return psi, omega, phi
    except Exception as err:
        if len(backbone) != backbone.seq_length()*3:
            raise BadStructureError("AtomArray has insufficient amount"
                "of backbone atoms") from None
        else:
            raise