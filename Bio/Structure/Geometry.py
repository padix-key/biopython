# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import Atom, AtomArray, AtomArrayStack
from . import position

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
    if type(atoms) == Atom:
        return atoms.pos
    if type(atoms) == AtomArray:
        return np.mean(atoms.pos, axis=0)
    if type(atoms) == AtomArrayStack:
        return np.mean(atoms.pos, axis=1)


def angle(atom1, atom2, atom3):
    pass


def dihedral(atom1, atom2, atom3, atom4):
    v1 = position(atom2) - position(atom1)
    v2 = position(atom3) - position(atom2)
    v3 = position(atom4) - position(atom3)
    
    n1 = np.cross(v1, v2)
    n1 /= np.linalg.norm(n1, axis=-1)[:, np.newaxis]
    n2 = np.cross(v2, v3)
    n2 /= np.linalg.norm(n2, axis=-1)[:, np.newaxis]
    
    v2 /= np.linalg.norm(v2, axis=-1)[:, np.newaxis]
    
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    
    return np.arctan2(y,x)


def dihedral_backbone(atom_array, chain_id):
    backbone = atom_array[((atom_array.atom_name == "N") |
                          (atom_array.atom_name == "CA") |
                          (atom_array.atom_name == "C")) &
                          (atom_array.hetero == " ") &
                          (atom_array.chain_id == chain_id)]
    angle_atoms = np.zeros(( (backbone.seq_length(chain_id)-1)*3, 4, 3 ))
    for i in range(len(angle_atoms)):
        angle_atoms[i] = backbone.pos[0+i : 4+i]
    dihed = dihedral(angle_atoms[:,0], angle_atoms[:,1], angle_atoms[:,2], angle_atoms[:,3])
    print(dihed)
    psi = dihed[0::3]
    omega = dihed[1::3]
    phi = dihed[2::3]
    
    return psi, omega, phi


def vector_dot(v1,v2):
    return (v1*v2).sum(axis=-1)