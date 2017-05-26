# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module provides functions for calculation of characteristic values when
comparing multiple structures with each other.
"""

import numpy as np
from . import Atom, AtomArray, AtomArrayStack
from . import vector_dot

def rmsd(reference, subject):
    """
    Calculate the RMSD between two structures.
    
    Calculate the root-mean-square-deviation (RMSD)
    of a structure compared to a reference structure.
    The RMSD is defined as:
    
    .. math:: RMSD = \\sqrt{ \\frac{1}{n} \\sum\\limits_{i=1}^n (x_i - x_{ref,i})^2}
    
    Parameters
    ----------
    reference : AtomArray
        Reference structure.
    subject : AtomArray or AtomArrayStack
        Structure(s) to be compared with `reference`.
        `reference` and `subject` must have equal annotation arrays.
    
    Returns
    -------
    rmsd : float or 1-D ndarray
        RMSD between subject and reference.
        If subject is an `AtomArray` a float is returned.
        If subject is an `AtomArrayStack` an `ndarray`
        containing the RMSD for each `AtomArray` is returned.
    
    See Also
    --------
    rmsf
    """
    sq_euclidian = _sq_euclidian(reference, subject)
    return np.sqrt(np.mean(sq_euclidian, axis=-1))


def rmsf(reference, subject):
    """
    Calculate the RMSF between two structures.
    
    Calculate the root-mean-square-fluctuation (RMSF)
    of a structure compared to a reference structure.
    The RMSF is defined as:
    
    .. math:: RMSF(i) = \\sqrt{ \\frac{1}{T} \\sum\\limits_{t=1}^T (x_i(t) - x_{ref,i}(t))^2}
    
    Parameters
    ----------
    reference : AtomArray
        Reference structure.
    subject : AtomArrayStack
        Structures to be compared with `reference`.
        reference` and `subject` must have equal annotation arrays.
        The time `t` is represented by the index of the first dimension
        of the AtomArrayStack.
    
    Returns
    -------
    rmsf : 1-D ndarray
        RMSF between subject and reference structure.
        The index corresponds to the atoms in the annotation arrays.
    
    See Also
    --------
    rmsd
    """
    if type(subject) != AtomArrayStack:
        raise ValueError("Subject must be AtomArrayStack")
    sq_euclidian = _sq_euclidian(reference, subject)
    return np.sqrt(np.mean(sq_euclidian, axis=0))
    np.linalg.svd(a)


def average(atom_arrays):
    """
    Calculate an average structure
    
    Calculate the average structure by calculating the average coordinates
    of each atom.
    
    Parameters
    ----------
    atom_arrays : AtomArrayStack
        Stack of structures to be averaged
    
    Returns
    -------
    average : AtomArray
        Structure with averaged atom coordinates.
    
    See Also
    --------
    rmsd, rmsf
    
    Notes
    -----
    The calculated average structure is not suitable for visualisation
    or geometric calculations, since bond lengths and angles will
    deviate from meaningful values.
    This method is rather useful to provide a reference structure for
    calculation of e.g. the RMSD or RMSF. 
    """
    mean_array = atom_arrays[0].copy()
    mean_array.coord = np.mean(atom_arrays.coord, axis=0)
    return mean_array


def _sq_euclidian(reference, subject):
    """
    Calculate squared euclidian distance between atoms in two structures.
    
    Parameters
    ----------
    reference : AtomArray
        Reference structure.
    subject : AtomArray or AtomArrayStack
        Structure(s) whose atoms squared euclidian distance to `reference`
        is measured.
    
    Returns
    -------
    1-D ndarray or 2-D ndarray
        Squared euclidian distance between subject and reference.
        If subject is an `AtomArray` a 1-D array is returned.
        If subject is an `AtomArrayStack` a 2-D array is returned.
        In this case the first dimension indexes the AtomArray.
    """
    if type(reference) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    dif = subject.coord - reference.coord
    return vector_dot(dif, dif)