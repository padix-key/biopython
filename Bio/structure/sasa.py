# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
Use this module to calculate the Solvent Accessible Surface Area (SASA) of
a protein or single atoms.
"""

import numpy as np
from . import AdjacencyMap


def sasa(array, **kwargs):
    """
    array : AtomArray
    probe_radius : float
    relevant_indices : ndarray, optional
    ignore_ions : bool
    point_number : int, optional
    point_distr : string or function, optional
    vdw_radii : string or ndarray, optional
    """
    if "probe_radius" in kwargs:
        probe_radius = float(kwargs["probe_radius"])
    else:
        probe_radius = 1.4
    
    if "relevant_indices" in kwargs:
        indices = np.array(kwargs["relevant_indices"])
    else:
        indices = np.arange(len(array))
    
    # Remove water residues, since it is the solvent
    array = array[array.hetero != "W"]
    if "ignore_ions" in kwargs:
        ignore_ions = np.array(kwargs["ignore_ions"])
    else:
        ignore_ions = True
    if ignore_ions:
        pass
    
    if "point_number" in kwargs:
        point_number = int(kwargs["point_number"])
    else:
        point_number = 100
    if "point_distr" in kwargs:
        point_distr = kwargs["point_distr"]
    else:
        point_distr = "Fibonacci"
    if type(point_distr) == function:
        sphere_points = function(point_number)
    elif point_distr == "Fibonacci":
        sphere_points = _create_fibonacci_points(point_number)
    else:
        raise ValueError("'" + str(point_distr) +
                         "' is not a valid point distribution")
    
    if "vdw_radii" in kwargs:
        vdw_radii = kwargs["vdw_radii"]
    else:
        vdw_radii = "ProtOr"
    if instanceof(vdw_radii, ndarray):
        radii = vdw_radii
        if len(radii) != len(array):
            raise ValueError("VdW radii array contains insufficient"
                             "amount of elements")
    elif vdw_radii == "ProtOr":
        if (array.element == "H").any():
            raise ValueError("ProtOr cannot be used for"
                             "structures with hydrogen atoms")
        array = array[array.element != "H"]
        radii = np.zeros(len(array))
        for i in range(len(radii)):
            try:
                radii[i] = _protor_radii[array.res_name[i]]
                                        [array.atom_name[i]]
            except KexError:
                radii[i] = _protor_default
    elif vdw_radii == "Single":
        radii = np.zeros(len(array))
        for i in range(len(radii)):
            try:
                radii[i] = _single_radii[array.element[i]]
            except KexError:
                radii[i] = _protor_default
    radii += probe_radius
    
    # Box size is as large as the maximum distance, 
    # where two atom can intersect.
    # Therefore intersecting atoms are always in the same or adjacent box.
    adj_map = AdjacencyMap(array, np.max(radii)*2)


def _create_fibonacci_points(n):
    """
    Get an array of approximately equidistant points on a sphere surface
    using a golden section spiral.
    """
    phi = (3 - np.sqrt(5)) * np.pi * np.arange(n)
    z = np.linspace(1 - 1.0/n, 1.0/n - 1, n)
    radius = np.sqrt(1 - z*z)
    coords = np.zeros((n, 3))
    coords[:,0] = radius * np.cos(phi)
    coords[:,1] = radius * np.sin(phi)
    coords[:,2] = z
    return coords


_protor_default = 1.80
C3H0s = 1.61
C3H0b = 1.61
C4H1s = 1.88
C4H1b = 1.88
C3H1s = 1.76
C3H1b = 1.76
C3H2u = 1.76
C4H2s = 1.88
C4H2b = 1.88
C4H3u = 1.88
N3H0u = 1.64
N3H1s = 1.64
N3H1b = 1.64
N4H3u = 1.64
N3H2u = 1.64
O1H0u = 1.42
O2H1u = 1.46
O2H2u = 1.46
S2H0u = 1.77
S2H1u = 1.77
_protor_radii = {"GLY": {" O  ": O1H0u,
                         " C  ": C3H0b,
                         " CA ": C4H2s,
                         " N  ": N3H1s},
                 "ALA": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1b,
                         " N  ": N3H1s,
                         " CB ": C4H3u},
                 "VAL": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H1b,
                         " CG1": C4H3u,
                         " CG2": C4H3u},
                 "LEU": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C4H1b,
                         " CD1": C4H3u,
                         " CD2": C4H3u},
                 "ILE": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H1b,
                         " CG1": C4H2b,
                         " CG2": C4H3u,
                         " CD1": C4H3u},
                 "PRO": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1b,
                         " N  ": N3H0u,
                         " CB ": C4H2b,
                         " CG ": C4H2b,
                         " CD ": C4H2s},
                 "MET": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C4H2b,
                         " SD ": S2H0u,
                         " CE ": C4H3u},
                 "PHE": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C3H0b,
                         " CD1": C3H1s,
                         " CD2": C3H1b,
                         " CE1": C3H1b,
                         " CE2": C3H1b,
                         " CZ ": C3H1b},
                 "TYR": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C3H0b,
                         " CD1": C3H1s,
                         " CD2": C3H1s,
                         " CE1": C3H1s,
                         " CE2": C3H1s,
                         " CZ ": C3H0b,
                         " OH ": O2H1u},
                 "TRP": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C3H0b,
                         " CD1": C3H1s,
                         " NE1": N3H1b,
                         " CD2": C3H0b,
                         " CE2": C3H0b,
                         " CE3": C3H1b,
                         " CZ3": C3H1b,
                         " CZ2": C3H1b,
                         " CH2": C3H1b,
                         "CEH2": C3H1b},
                 "SER": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " OG ": O2H1u,
                         " OG1": O2H1u},
                 "THR": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H1b,
                         " OG1": O2H1u,
                         " CG2": C4H3u,
                         " CG ": C4H3u},
                 "ASN": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C3H0b,
                         " OD1": O1H0u,
                         " ND2": N3H2u},
                 "GLN": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C4H2s,
                         " CD ": C3H0b,
                         " OE1": O1H0u,
                         " NE2": N3H2u},
                 "CYS": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2b,
                         " SG ": S2H1u},
                 "CSS": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2b,
                         " SG ": S2H0u},
                 "HIS": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C3H0b,
                         " ND1": N3H1b,
                         " CD2": C3H1s,
                         " CE1": C3H1s,
                         " NE2": N3H1b},
                 "GLU": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C4H2s,
                         " CD ": C3H0b,
                         " OE1": O1H0u,
                         " OE2": O1H0u},
                 "ASP": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C3H0b,
                         " OD1": O1H0u,
                         " OD2": O1H0u},
                 "ARG": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C4H2s,
                         " CD ": C4H2s,
                         " NE ": N3H1b,
                         " CZ ": C3H0b,
                         " NH1": N3H2u,
                         " NH2": N3H2u},
                 "LYS": {" O  ": O1H0u,
                         " C  ": C3H0s,
                         " CA ": C4H1s,
                         " N  ": N3H1s,
                         " CB ": C4H2s,
                         " CG ": C4H2s,
                         " CD ": C4H2s,
                         " CE ": C4H2b,
                         " NZ ": N4H3u}}


_single_radii = {"H":  1.20,
                 "C":  1.70,
                 "N":  1.55,
                 "O":  1.52,
                 "F":  1.47,
                 "Si": 2.10,
                 "P":  1.80,
                 "S":  1.80,
                 "Cl": 1.75,
                 "Br": 1.85,
                 "I":  1.98}