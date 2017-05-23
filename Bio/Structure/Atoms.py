# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
import Bio.PDB

class _AtomAnnotationList(object):
    
    def __init__(self, length: int=None):
        if length == None:
            return
        self.chain_id = np.zeros(length, dtype="U1")
        self.res_id = np.zeros(length, dtype=int)
        self.res_name = np.zeros(length, dtype="U3")
        self.atom_name = np.zeros(length, dtype="U4")
        self.hetero = np.zeros(length, dtype="U3")
        
    def seq_length(self, chain_id: str="all"):
        if chain_id != "all":
            res_id_by_chain = self.res_id[self.chain_id == chain_id]
        else:
            res_id_by_chain = self.res_id
        # Count the number of times a new res_id is found
        last_found_id = -1
        id_count = 0
        i = 0
        while i < len(res_id_by_chain):
            found_id = res_id_by_chain[i]
            if last_found_id != found_id:
                last_found_id = found_id
                id_count += 1
            i += 1
        return id_count
        
    def check_integrity(self):
        if self.chain_id.shape != (len(self),):
            return False
        if self.res_id.shape != (len(self)):
            return False
        if self.res_name.shape != (len(self)):
            return False
        if self.atom_name.shape != (len(self)):
            return False
        if self.hetero.shape != (len(self)):
            return False
        return True
    
    def equal_annotations(self):
        if not isinstance(item, _AtomAnnotationList):
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
    
    def __eq__(self, item):
        return self.equal_annotations()
    
    def __ne__(self, item):
        return not self.__eq__(item)
    
    def __len__(self):
        # length is determined by length of chain_id attribute
        return self.chain_id.shape[0]
    

class Atom(object):
    
    def __init__(self, chain_id: str, res_id: int, res_name: str,
                 atom_name: str, hetero: str="", pos=np.zeros(3)):
        self.chain_id = chain_id
        self.res_id = res_id
        self.res_name = res_name
        self.atom_name = atom_name
        self.hetero = hetero
        pos = np.array(pos, dtype=float)
        # Check if pos contains x,y and z coordinates
        if pos.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.pos = pos
    
    def __str__(self):
        return (self.chain_id + "\t" + str(self.res_id) + "\t" +
                self.res_name + "\t" + self.atom_name + "\t" + 
                self.hetero + "\t" + str(self.pos))

    
class AtomArray(_AtomAnnotationList):
    
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
        return new_array
        
    def check_integrity(self):
        if not super().check_integrity():
            return False
        if self.pos.shape != (len(self), 3):
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
        while i < len(self):
            yield self.get_atom(i)
            i += 1
    
    def __getitem__(self, index):
        try:
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
                return new_array
        except:
            raise IndexError("Invalid index") from None
        
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
        else:
            raise IndexError("Index must be integer")
        
    def __len__(self):
        # length is determined by length of pos attribute
        return self.pos.shape[0]
    
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


class AtomArrayStack(_AtomAnnotationList):
    
    def __init__(self, depth: int=None, length: int=None):
        if depth == None or length == None:
            return
        super().__init__(length)
        self.pos = np.zeros((depth, length, 3), dtype=float)
    
    def check_integrity(self):
        if not super().check_integrity():
            return False
        if self.pos.shape != (len(self), super().__len__(), 3):
            return False
        return True
    
    def get_array(self, index):
        array = AtomArray()
        array.chain_id = self.chain_id
        array.res_id = self.res_id
        array.res_name = self.res_name
        array.atom_name = self.atom_name
        array.hetero = self.hetero
        array.pos = self.pos[index]
        return array

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self.get_array(i)
            i += 1
            
    def __getitem__(self, index):
        try:
            if isinstance(index, int):
                return self.get_array(index)
            elif isinstance(index, tuple):
                new_stack = AtomArrayStack()
                new_stack.chain_id = self.chain_id.__getitem__(index[1:])
                new_stack.res_id = self.res_id.__getitem__(index[1:])
                new_stack.res_name = self.res_name.__getitem__(index[1:])
                new_stack.atom_name = self.atom_name.__getitem__(index[1:])
                new_stack.hetero = self.hetero.__getitem__(index[1:])
                new_stack.pos = self.pos.__getitem__(index)
                return new_stack
            else:
                new_stack = AtomArrayStack()
                new_stack.chain_id = self.chain_id.__getitem__(index)
                new_stack.res_id = self.res_id.__getitem__(index)
                new_stack.res_name = self.res_name.__getitem__(index)
                new_stack.atom_name = self.atom_name.__getitem__(index)
                new_stack.hetero = self.hetero.__getitem__(index)
                new_stack.pos = self.pos.__getitem__(index)
                return new_stack
        except:
            raise IndexError("Invalid index")
            
    
    def __setitem__(self, index: int, array: AtomArray):
        if not super(AtomArray, array).__eq__(array):
            raise ValueError("The array's atom annotations do not fit")
        if isinstance(index, int):
            self.pos[index] = array.pos
        else:
            raise IndexError("Index must be integer")
        
    def __delitem__(self, index: int):
        if isinstance(index, int):
            self.pos = np.delete(self.pos, index, axis=0)
        else:
            raise IndexError("Index must be integer")
    
    def __len__(self):
        # length is determined by length of pos attribute
        return self.pos.shape[0]
    
    def __eq__(self, item):
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArrayStack):
            return False
        if not np.array_equal(self.pos, item.pos):
            return False
        return True
    
    def __ne__(self, item):
        return not self.__eq__(item)
    
    def __str__(self):
        string = ""
        for i, array in enumerate(self):
            string += "Model: " + str(i) + "\n"
            string += str(array) + "\n" + "\n"
        return string


def stack(arrays):
    for array in arrays:
        # Check if all arrays share equal annotations
        if not super(AtomArray, array).__eq__(arrays[0]):
            raise ValueError("The arrays atom annotations do not fit to each other") 
    array_stack = AtomArrayStack()
    array_stack.chain_id = arrays[0].chain_id
    array_stack.res_id = arrays[0].res_id
    array_stack.res_name = arrays[0].res_name
    array_stack.atom_name = arrays[0].atom_name
    array_stack.hetero = arrays[0].hetero
    pos_list = [array.pos for array in arrays] 
    array_stack.pos = np.stack(pos_list, axis=0)
    return array_stack

def to_array(model: Bio.PDB.Model.Model, insertion_code: str=""):
    arr = AtomArray(_get_model_size(model))
    i = 0
    for chain in model:
        for residue in chain:
            # Only recognize atoms with given insertion code
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
    model = Bio.PDB.Model.Model(0)
    # Iterate through all atoms
    for i in range(len(array)):
        # Extract annotations and position of every atom
        chain_id = array.chain_id[i]
        hetero = array.hetero[i]
        res_id = array.res_id[i]
        res_name = array.res_name[i]
        atom_name = array.atom_name[i]
        pos = array.pos[i]
        # Try to access the chain entity that corresponds to this atom
        # if chain does not exist create chain and add it to super entity (model)
        try:
            chain_curr = model[chain_id]
        except KeyError:
            chain_curr = Bio.PDB.Chain.Chain(chain_id)
            model.add(chain_curr)
        # Same as above with residues
        try:
            res_curr = chain_curr[(hetero, res_id, " ")]
        except KeyError:
            res_curr = Bio.PDB.Residue.Residue(
                (hetero, res_id, " "), res_name, " ")
            chain_curr.add(res_curr)
        # Same as above with atoms
        try:
            atom_curr = res_curr[atom_name]
        except KeyError:
            atom_curr = Bio.PDB.Atom.Atom(atom_name, pos, 0, 1, " ",
                                          atom_name, i+1, atom_name[0])
            res_curr.add(atom_curr)
    return model


def _get_model_size(model: Bio.PDB.Model.Model, insertion_code: str=""):
    size = 0
    for chain in model:
        for residue in chain:
            # Only recognize atoms with given insertion code
            insertion = _get_insertion_code(residue)
            if insertion == insertion_code:
                for atom in residue:
                    size += 1
    return size


def _get_insertion_code(residue: Bio.PDB.Residue.Residue):
    return residue.id[2].strip()


def position(item):
    if type(item) in (Atom, AtomArray, AtomArrayStack):
        return item.pos
    else:
        return np.array(item)