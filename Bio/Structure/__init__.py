# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""Provides objects for handling protein structure data.


"""

import numpy as np
import Bio.PDB

class _AtomPropertyList(object):
    
    def __init__(self, length: int=None):
        if length == None:
            return
        self.chain_id = np.zeros(length, dtype="U1")
        self.res_id = np.zeros(length, dtype=int)
        self.res_name = np.zeros(length, dtype="U3")
        self.atom_name = np.zeros(length, dtype="U4")
        self.hetero = np.zeros(length, dtype="U3")
        self._length = length
        
    def check_integrity(self):
        if self.chain_id.shape != (self._length,):
            return False
        if self.res_id.shape != (self._length,):
            return False
        if self.res_name.shape != (self._length,):
            return False
        if self.atom_name.shape != (self._length,):
            return False
        if self.hetero.shape != (self._length,):
            return False
        return True
    
    def __eq__(self, item):
        if not isinstance(item, _AtomPropertyList):
            return False
        if not np.array_equal(self.chain_id, item.chain_id):
            return False
        if not np.array_equal(self.res_id, item.res_id):
            return False
        if not np.array_equal(self.res_name, item.res_name):
            return False
        if not np.array_equal(self.atom_name, item.atom_name):
            return False
        if not np.array_equal(self.hetero, item.hetero):
            return False
        return True
    
    def __ne__(self, item):
        return not self.__eq__(item)
    

class Atom(object):
    
    def __init__(self, chain_id: str, res_id: int, res_name: str,
                 atom_name: str, hetero: str="", pos=np.zeros(3)):
        self.chain_id = chain_id
        self.res_id = res_id
        self.res_name = res_name
        self.atom_name = atom_name
        self.hetero = hetero
        pos = np.array(pos, dtype=float)
        if pos.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.pos = pos
    
    def __str__(self):
        return (self.chain_id + "\t" + str(self.res_id) + "\t" +
                self.res_name + "\t" + self.atom_name + "\t" + 
                self.hetero + "\t" + str(self.pos))

    
class AtomArray(_AtomPropertyList):
    
    def __init__(self, length: int=None):
        if length == None:
            return
        super().__init__(length)
        self.pos = np.zeros((length, 3), dtype=float)
        
    def copy(self):
        new_array = AtomArray()
        new_array.chain_id = np.copy(self.chain_id)
        new_array.res_id = np.copy(self.res_id)
        new_array.res_name = np.copy(self.res_name)
        new_array.atom_name = np.copy(self.atom_name)
        new_array.hetero = np.copy(self.hetero)
        new_array.pos = np.copy(self.pos)
        new_array._length = self._length
        return new_array
        
    def sort(self):
        pass
        
    def check_integrity(self):
        if not super().check_integrity():
            return False
        if self.pos.shape != (self._length, 3):
            return False
        return True
    
    def get_atom(self, index):
        return Atom(chain_id = self.chain_id[index],
                    res_id = self.res_id[index],
                    res_name = self.res_name[index],
                    atom_name = self.atom_name[index],
                    hetero = self.hetero[index],
                    pos = self.pos[index])
    
    def __iter__(self):
        i = 0
        while i < self._length:
            yield self.get_atom(i)
            i += 1
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_atom(index)
        else:
            new_array = AtomArray()
            new_array.chain_id = self.chain_id.__getitem__(index)
            new_array.res_id = self.res_id.__getitem__(index)
            new_array.res_name = self.res_name.__getitem__(index)
            new_array.atom_name = self.atom_name.__getitem__(index)
            new_array.hetero = self.hetero.__getitem__(index)
            new_array.pos = self.pos.__getitem__(index)
            new_array._length = len(new_array.pos)
            if not new_array.check_integrity():
                raise IndexError("Index created invalid AtomArray")
            return new_array
        
    def __setitem__(self, index: int, atom: Atom):
        if isinstance(index, int):
            self.chain_id[index] = atom.chain_id
            self.res_id[index] = atom.res_id
            self.res_name[index] = atom.res_name
            self.atom_name[index] = atom.atom_name
            self.hetero[index] = atom.hetero
            self.pos[index] = atom.pos
        else:
            raise IndexError("Index must be integer")
        
    def __delitem__(self, index: int):
        if isinstance(index, int):
            self.chain_id = np.delete(self.chain_id, index, axis=0)
            self.res_id = np.delete(self.res_id, index, axis=0)
            self.res_name = np.delete(self.res_name, index, axis=0)
            self.atom_name = np.delete(self.atom_name, index, axis=0)
            self.hetero = np.delete(self.hetero, index, axis=0)
            self.pos = np.delete(self.pos, index, axis=0)
            self._length -= 1
        else:
            raise IndexError("Index must be integer")
        
    def __len__(self):
        return self._length
    
    def __eq__(self, item):
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArray):
            return False
        if not np.array_equal(self.pos, item.pos):
            return False
        return True
    
    def __ne__(self, item):
        return not self.__eq__(item)
    
    def __str__(self):
        string = ""
        for atom in self:
            string += str(atom) + "\n"
        return string


class AtomArrayStack(_AtomPropertyList):
    
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
    
    def get_array(self, index):
        pass
    
    def __iter__(self):
        i = 0
        while i < self._depth:
            yield self.get_array(i)
            i += 1
            
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return self._depth
    
    def __eq__(self, item):
        pass
    
    def __ne__(self, item):
        return not self.__eq__(item)
    
    def __str__(self):
        pass


def to_array(model: Bio.PDB.Model.Model, insertion_code: str=""):
    arr = AtomArray(_get_model_length(model))
    i = 0
    for chain in model:
        for residue in chain:
            insertion = _get_insertion_code(residue)
            if insertion == insertion_code:
                for atom in residue:
                    arr.chain_id[i] = chain.id
                    arr.hetero[i] = residue.id[0]
                    arr.res_id[i] = int(residue.id[1])
                    arr.res_name[i] = residue.get_resname()
                    arr.atom_name[i] = atom.get_id()
                    arr.pos[i] = atom.get_coord()
                    i += 1
    return arr


def to_model(array: AtomArray):
    pass


def _get_model_length(model: Bio.PDB.Model.Model, insertion_code: str=""):
    length = 0
    for chain in model:
        for residue in chain:
            insertion = _get_insertion_code(residue)
            if insertion == insertion_code:
                for atom in residue:
                    length += 1
    return length


def _get_insertion_code(residue: Bio.PDB.Residue.Residue):
    return residue.id[2].strip()

def ensure_structure_type(item, allow_single=True, allow_array=True, allow_stack=True):
    if isinstance(item, Atom):
        if allow_single:
            return "single"
        else:
            raise ValueError("Object cannot be a single atom")
    elif isinstance(item, AtomArray):
        if allow_array:
            return "array"
        else:
            raise ValueError("Object cannot be an atom array")
    elif isinstance(item, AtomArrayStack):
        if allow_stack:
            return "stack"
        else:
            raise ValueError("Object cannot be a atom array stack")
    else:
        raise ValueError("Object is not a structure type")