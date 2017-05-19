# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from .. import *
from ..Measure import rmsd, get_centroid

def superimpose(reference, subject, fast=True):
    if not isinstance(reference, AtomArray):
        raise ValueError("Reference must be AtomArray")
    struc_type = ensure_structure_type(subject, allow_single=False)
    
    if struc_type == "array":
        sub_centroid = get_centroid(subject)
        ref_centroid = get_centroid(reference)
        if fast:
            # For performance reasons the Kabsch algorithm
            # is only performed with "CA"
            # Implicitly this creates array copies
            sub_centered = subject[(subject.atom_name == "CA")]
            ref_centered = reference[(reference.atom_name == "CA")]
        else:
            sub_centered = subject.copy();
            ref_centered = reference.copy();
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
        
        if fast:
            fitted_subject = subject.copy()
            fitted_subject.pos -= sub_centroid
        else:
            fitted_subject = sub_centered
        fitted_subject.pos = np.dot(fitted_subject.pos, rotation)
        fitted_subject.pos += ref_centroid
        
        return fitted_subject, rotation, ref_centroid, sub_centroid
    
    else:
        pass
        # TODO