# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module allows efficient search of atoms in a defined radius around
a location.
"""

import numpy as np
from . import distance

class AdjacencyMap(object):
    
    def __init__(self, atom_array, box_size):
        self.array = atom_array.copy()
        self.boxsize = box_size
        # calculate how many boxes are required for each dimension
        self.min_coord = np.min(self.array.coord, axis=-2)
        self.max_coord = np.max(self.array.coord, axis=-2)
        self.box_count = ((((self.max_coord-self.min_coord) / box_size)+1)
                          .astype(int))
        self.boxes = np.zeros(self.box_count, dtype=object)
        # Fill boxes with empty lists, cannot use ndarray.fill(),
        # since it fills the entire array with the same list instance
        for x in range(self.boxes.shape[0]):
            for y in range(self.boxes.shape[1]):
                for z in range(self.boxes.shape[2]):
                    self.boxes[x,y,z] = []
        for i, pos in enumerate(self.array.coord):
            self.boxes[self._get_box_index(pos)].append(i)
    
    def get_atoms(self, coord, radius):
        indices = self.get_atoms_in_box(coord, int(radius/self.boxsize)+1)
        sel_coord = self.array.coord[indices]
        dist = distance(sel_coord, coord)
        return indices[dist <= radius]
    
    def get_atoms_in_box(self, coord, box_r=1):
        box_i =  self._get_box_index(coord)
        atom_indices = []
        shape = self.boxes.shape
        for x in range(box_i[0]-box_r, box_i[0]+box_r+1):
            if (x >= 0 and x < shape[0]):
                for y in range(box_i[1]-box_r, box_i[1]+box_r+1):
                    if (y >= 0 and y < shape[1]):
                        for z in range(box_i[2]-box_r, box_i[2]+box_r+1):
                            if (z >= 0 and z < shape[2]):
                                atom_indices.extend(self.boxes[x,y,z])
        return np.array(atom_indices)
    
    def get_atom_array(self):
        return self.array
    
    def _get_box_index(self, coord):
        return tuple(((coord-self.min_coord) / self.boxsize).astype(int))
    
    def __str__(self):
        return(self.boxes)