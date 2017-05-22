# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import centroid
from . import Atom, AtomArray, AtomArrayStack

def superimpose(reference, subject, ca_only=True):
    if type(reference) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    
    if type(subject) == AtomArray:
        sub_centroid = centroid(subject)
        ref_centroid = centroid(reference)
        if ca_only:
            # For performance reasons the Kabsch algorithm
            # is only performed with "CA"
            # Implicitly this creates array copies
            sub_centered = subject[(subject.atom_name == "CA")]
            ref_centered = reference[(reference.atom_name == "CA")]
        else:
            sub_centered = subject.copy();
            ref_centered = reference.copy();
            
        if len(sub_centered) != len(ref_centered):
            raise BadStructureException("The subject and reference array have different amount of atoms")
        
        sub_centered.pos -= sub_centroid
        ref_centered.pos -= ref_centroid
        
        # Calculating rotation matrix using Kabsch algorithm
        y = sub_centered.pos
        x = ref_centered.pos
        cov = np.dot(y.T, x)
        v, s, w = np.linalg.svd(cov)
        rotation = np.dot(w, v.T)
        if np.linalg.det(v) * np.linalg.det(w) < 0:
            s[-1,:] *= -1
            v[:,-1] *= -1
        rotation = np.dot(v,w)
        
        if ca_only:
            fitted_subject = subject.copy()
            fitted_subject.pos -= sub_centroid
        else:
            fitted_subject = sub_centered
        fitted_subject.pos = np.dot(fitted_subject.pos, rotation)
        fitted_subject.pos += ref_centroid
        
        return fitted_subject, (sub_centroid,rotation,ref_centroid)
    
    elif type(subject) == AtomArray:
        pass
        # TODO
        
    else:
        raise ValueError("Subject must be AtomArray or AtomArrayStack")


def apply_superimposition(atoms, transformation):
    transformed = atoms.copy()
    transformed.pos -= transformation[0]
    transformed.pos = np.dot(transformed.pos, transformation[1])
    transformed.pos += transformation[2]