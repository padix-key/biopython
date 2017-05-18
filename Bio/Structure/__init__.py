# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""Provides objects for handling protein structure data.


"""

import numpy as np
import Bio.PDB

class AtomPropertyList(object):
    
    def __init__(self, length: int):
        self.chain = np.zeros(length, dtype=str)
        self.res_id = np.zeros(length, dtype=int)
        self.res_name = np.zeros(length, dtype=str)
        self.atom_name = np.zeros(length, dtype=str)
        self.hetfield = np.zeros(length, dtype=str)
        self._length = length
        
    def check_integrity(self):
        if self.chain.shape != (self._length,):
            return False
        if self.res_id.shape != (self._length,):
            return False
        if self.res_name.shape != (self._length,):
            return False
        if self.atom_name.shape != (self._length,):
            return False
        if self.hetfield.shape != (self._length,):
            return False
        return True


class Atom(object):
    
    def __init__(self, chain: str, res_id: int, res_name: str, atom_name: str, hetfield: str="", pos=np.zeros(3)):
        self.chain = chain
        self.res_id = res_id
        self.res_name = res_name
        self.atom_name = atom_name
        self.hetfield = hetfield
        pos = np.array(pos, dtype=float)
        if pos.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.pos = pos

    
class AtomArray(AtomPropertyList):
    
    def __init__(self, length):
        super().__init__(length)
        self.pos = np.zeros((length, 3), dtype=float)
        
    def check_integrity(self):
        if not super().check_integrity():
            return False
        if self.pos.shape != (self._length, 3):
            return False
        return True


class AtomArrayStack(AtomPropertyList):
    
    def __init__(self, depth: int, length: int):
        super().__init__(length)
        self._depth = depth
        self.pos = np.zeros((depth, length, 3), dtype=float)
    
    def check_integrity(self):
        if not super().check_integrity():
            return False
        if self.pos.shape != (self._depth, self._length, 3):
            return False
        return True


def to_array(model: Bio.PDB.Model, insertion_code: str=""):
    pass


def to_model(array: AtomArray):
    pass


def _get_model_length(model):
    pass
