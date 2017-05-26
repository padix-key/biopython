# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module contains the main types of the `Structure` subpackage: `Atom`,
`AtomArray` and `AtomArrayStack`.

In this context an atom is described by two kinds of attributes: the position
and the annotations. The annotations include information about polypetide chain
id, residue id, residue name, hetero atom information and atom name. A position
is a `numpy` float ndarray of length 3, containing the x, y and z coordinates.

An `Atom` contains data for a single atom, it stores the annotations as scalar
values and the position as length 3 ndarray.
An `AtomArray` stores data for an entire model containing *n* atoms.
Therefore the annotations are represented as ndarrays of length *n*, so called
annotation arrays. The position is a (n x 3) ndarray.
`AtomArrayStack` stores data for *m* models. For this, each `AtomArray` in
the `AtomArrayStack` has the same annotation arrays, but may differ in atom
positions. Therefore the annotation arrays are represented as ndarrays of
length *n*, while the position is (m x n x 3) ndarray.
All types must not be subclassed.

For each type, the attributes can be accessed directly. Both `AtomArray` and
`AtomArrayStack` support `numpy` style indexing, the index is propagated to
each attribute. If a single integer is used as index, an object with one
dimension less is returned
(`AtomArrayStack` -> `AtomArray`, `AtomArray` -> `Atom`).
"""

import numpy as np
import Bio.PDB

class _AtomAnnotationList(object):
    """
    Representation of the annotation arrays for
    `AtomArray` and `AtomArrayStack`.
    """
    
    def __init__(self, length: int=None):
        """
        Create the annotation arrays
        """
        if length == None:
            return
        # string size based on reserved columns in *.pdb files
        self.chain_id = np.zeros(length, dtype="U1")
        self.res_id = np.zeros(length, dtype=int)
        self.res_name = np.zeros(length, dtype="U3")
        self.atom_name = np.zeros(length, dtype="U4")
        self.hetero = np.zeros(length, dtype="U5")
        
    def seq_length(self, chain_id: str="all"):
        """
        Calculate the amount of residues in a polypeptide chain.
        
        This is a quite expensive operation, since the function iterates
        through the `res_id` annotation array The amount of residues is
        determined by the amount of times a new a new `res_id` is found.
        Hetero residues are also taken into account.
        
        Parameters
        ----------
        chain_id : string, optional
            The polypeptide chain id, where the amount of residues is
            determined. The default value is `all`, where no filtering is
            applied.
        
        Returns
        -------
        length : int
            number of residues in the given chain.
        """
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
        """
        Check, if all annotation ndarrays have appropriate shapes.
        
        Returns
        -------
        integrity : boolean
            True, if the attribute shapes are consistent.
        """
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
    
    def equal_annotations(self, item):
        """
        Check, if this object shares equal annotation arrays with the given
        `AtomArray` or `AtomArrayStack`.
        
        Parameters
        ----------
        item : AtomArray or AtomArrayStack
            The object to compare the annotation arrays with.
        
        Returns
        -------
        equality : boolean
            True, if the aannotation arrays are equal.
        """
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
        """
        See Also
        --------
        equal_annotations
        """
        return self.equal_annotations()
    
    def __ne__(self, item):
        """
        See Also
        --------
        equal_annotations
        """
        return not self.__eq__(item)
    
    def __len__(self):
        """
        The length of the annotation arrays.
        
        Returns
        -------
        length : int
            Length of the annotation arrays.
        """
        # length is determined by length of chain_id attribute
        return self.chain_id.shape[0]
    

class Atom(object):
    """
    A representation of a single atom.
    
    All attributes correspond to `Entity` attributes in `Bio.PDB`.
    
    Attributes
    ----------
    chain_id : string {'A','B',...}
        A single character representing the polypeptide chain
    res_id : int
        A integer value identifying the sequence position of the residue
    res_name : string {'GLY','ALA',...}
        A three character string representing the residue name
    atom_name : string {' CA ',' N  ',...}
        A four character string representing the atom name.
        Pay attention to the whitespaces
    hetero : string {' ','W','H_GLC',...}
        An up to 5 character string, indicating in which hetero residue the
        atom is in. If the residue is a standard amino acid the value is `' '`.
    """
    
    def __init__(self, chain_id: str, res_id: int, res_name: str,
                 atom_name: str, hetero: str=" ", pos=np.zeros(3)):
        """
        Create an `Atom`.
        
        Parameters
        ----------
        See class attributes
        """
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
        """
        String representation of the atom.
        """
        return (self.chain_id + "\t" + str(self.res_id) + "\t" +
                self.res_name + "\t" + self.atom_name + "\t" + 
                self.hetero + "\t" + str(self.pos))

    
class AtomArray(_AtomAnnotationList):
    """
    An array representation of a structure consisting of multiple atoms.
    
    All attributes correspond to `Entity` attributes in `Bio.PDB`.
    
    Attributes
    ----------
    chain_id : ndarray(dtype="U1") {'A','B',...}
        A single character representing the polypeptide chain
    res_id : ndarray(dtype=int)
        A integer value identifying the sequence position of the residue
    res_name : ndarray(dtype="U3") {'GLY','ALA',...}
        A three character string representing the residue name
    atom_name : ndarray(dtype="U4") {' CA ',' N  ',...}
        A four character string representing the atom name.
        Pay attention to the whitespaces
    hetero : ndarray(dtype="U5") {' ','W','H_GLC',...}
        An up to 5 character string, indicating in which hetero residue the
        atom is in. If the residue is a standard amino acid the value is `' '`.
    pos : ndarray(dtype=float)
        (n x 3) ndarray containing the x, y and z coordinate of the atoms.
    """
    
    def __init__(self, length: int=None):
        """
        Create an `AtomArray`.
        
        Parameters
        ----------
        length : int, optional
            If length is given, the attribute arrays will be created with
            zeros.
        """
        if length == None:
            return
        super().__init__(length)
        self.pos = np.zeros((length, 3), dtype=float)
        
    def copy(self):
        """
        Create a new `AtomArray` instance with all attribute arrays copied.
        
        Returns
        -------
        new_array : AtomArray
            A deep copy of this array.
        """
        new_array = AtomArray()
        new_array.chain_id = np.copy(self.chain_id)
        new_array.res_id = np.copy(self.res_id)
        new_array.res_name = np.copy(self.res_name)
        new_array.atom_name = np.copy(self.atom_name)
        new_array.hetero = np.copy(self.hetero)
        new_array.pos = np.copy(self.pos)
        return new_array
        
    def check_integrity(self):
        """
        Check, if all attribute arrays have appropriate shapes.
        
        Returns
        -------
        integrity : boolean
            True, if the attribute shapes are consistent.
        """
        if not super().check_integrity():
            return False
        if self.pos.shape != (len(self), 3):
            return False
        return True
    
    def get_atom(self, index):
        """
        Obtain the atom instance of the array at the specified index.
        
        The same as ``array[index]``, if `index` is an integer.
        
        Parameters
        ----------
        index : int
            Index of the atom.
        
        Returns
        -------
        atom : Atom
            Atom at position `index`. 
        """
        return Atom(chain_id = self.chain_id[index],
                    res_id = self.res_id[index],
                    res_name = self.res_name[index],
                    atom_name = self.atom_name[index],
                    hetero = self.hetero[index],
                    pos = self.pos[index])
    
    def __iter__(self):
        """
        Iterate through the array.
        
        Yields
        ------
        atom : Atom
        """
        i = 0
        while i < len(self):
            yield self.get_atom(i)
            i += 1
    
    def __getitem__(self, index):
        """
        Obtain the atom instance or an subarray at the specified index.
        
        Parameters
        ----------
        index : object
            All index types `numpy` accepts are valid.
        
        Returns
        -------
        sub_array : Atom or AtomArray
            if `index` is an integer an `Atom` instance,
            otherwise an `AtomArray` with reduced length is returned.
        """
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
        """
        Set the atom at the specified array position.
        
        Parameters
        ----------
        index : int
            The position, where the atom is set.
        atom : Atom
            The atom to be set.
        """
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
        """
        Deletes the atom at the specified array position.
        
        Parameters
        ----------
        index : int
            The position where the atom should be deleted.
        """
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
        """
        The length of the array.
        
        Returns
        -------
        length : int
            Length of the array.
        """
        # length is determined by length of pos attribute
        return self.pos.shape[0]
    
    def __eq__(self, item):
        """
        Check if the array equals another `AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : boolean
            True, if `item` is an `AtomArray`
            and all its attribute arrays equals the ones of this object.
        """
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArray):
            return False
        if not np.array_equal(self.pos, item.pos):
            return False
        return True
    
    def __ne__(self, item):
        """
        See also
        --------
        __eq__
        """
        return not self.__eq__(item)
    
    def __str__(self):
        """
        Get a string representation of the array.
        
        Each line contains the attributes of one atom.
        """
        string = ""
        for atom in self:
            string += str(atom) + "\n"
        return string


class AtomArrayStack(_AtomAnnotationList):
    """
    A collection of multiple atom arrays, where each atom array has equal
    annotation arrays.
    
    Since the annotations are equal for each array the annotaion arrays are
    1-D, while the position array is 3-D (m x n x 3)-
    
    All attributes correspond to `Entity` attributes in `Bio.PDB`.
    
    Attributes
    ----------
    chain_id : ndarray(dtype="U1") {'A','B',...}
        A single character representing the polypeptide chain
    res_id : ndarray(dtype=int)
        A integer value identifying the sequence position of the residue
    res_name : ndarray(dtype="U3") {'GLY','ALA',...}
        A three character string representing the residue name
    atom_name : ndarray(dtype="U4") {' CA ',' N  ',...}
        A four character string representing the atom name.
        Pay attention to the whitespaces
    hetero : ndarray(dtype="U5") {' ','W','H_GLC',...}
        An up to 5 character string, indicating in which hetero residue the
        atom is in. If the residue is a standard amino acid the value is `' '`.
    pos : ndarray(dtype=float)
        (m x n x 3) ndarray containing the x, y and z coordinate of the atoms.
    """
    
    def __init__(self, depth: int=None, length: int=None):
        """
        Create an `AtomArrayStack`.
        
        Parameters
        ----------
        depth, length : int, optional
            If length and depth is given, the attribute arrays will be created
            with zeros. `depth` corresponds to the first dimension, `length`
            to the second.
        """
        if depth == None or length == None:
            return
        super().__init__(length)
        self.pos = np.zeros((depth, length, 3), dtype=float)
    
    def check_integrity(self):
        """
        Check, if all attribute arrays have appropriate shapes.
        
        Returns
        -------
        integrity : boolean
            True, if the attribute shapes are consistent.
        """
        if not super().check_integrity():
            return False
        if self.pos.shape != (len(self), super().__len__(), 3):
            return False
        return True
    
    def get_array(self, index):
        """
        Obtain the atom array instance of the stack at the specified index.
        
        The same as ``stack[index]``, if `index` is an integer.
        
        Parameters
        ----------
        index : int
            Index of the atom array.
        
        Returns
        -------
        array : AtomArray
            AtomArray at position `index`. 
        """
        array = AtomArray()
        array.chain_id = self.chain_id
        array.res_id = self.res_id
        array.res_name = self.res_name
        array.atom_name = self.atom_name
        array.hetero = self.hetero
        array.pos = self.pos[index]
        return array

    def __iter__(self):
        """
        Iterate through the array.
        
        Yields
        ------
        array : AtomArray
        """
        i = 0
        while i < len(self):
            yield self.get_array(i)
            i += 1
            
    def __getitem__(self, index):
        """
        Obtain the atom array instance or an substack at the specified index.
        
        Parameters
        ----------
        index : object
            All index types `numpy` accepts are valid.
        
        Returns
        -------
        sub_array : AtomArray or AtomArrayStack
            if `index` is an integer an `Atom` instance,
            otherwise an `AtomArray` with reduced length is returned.
        """
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
        """
        Set the atom array at the specified stack position.
        
        The array and the stack must have equal annotation arrays.
        
        Parameters
        ----------
        index : int
            The position, where the array atom is set.
        array : AtomArray
            The atom array to be set.
        """
        if not super(AtomArray, array).__eq__(array):
            raise ValueError("The array's atom annotations do not fit")
        if isinstance(index, int):
            self.pos[index] = array.pos
        else:
            raise IndexError("Index must be integer")
        
    def __delitem__(self, index: int):
        """
        Deletes the atom array at the specified stack position.
        
        Parameters
        ----------
        index : int
            The position where the atom array should be deleted.
        """
        if isinstance(index, int):
            self.pos = np.delete(self.pos, index, axis=0)
        else:
            raise IndexError("Index must be integer")
    
    def __len__(self):
        """
        The depth of the stack.
        
        Returns
        -------
        depth : int
            depth of the array.
        """
        # length is determined by length of pos attribute
        return self.pos.shape[0]
    
    def __eq__(self, item):
        """
        Check if the array equals another `AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : boolean
            True, if `item` is an `AtomArray`
            and all its attribute arrays equals the ones of this object.
        """
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArrayStack):
            return False
        if not np.array_equal(self.pos, item.pos):
            return False
        return True
    
    def __ne__(self, item):
        """
        See also
        --------
        __eq__
        """
        return not self.__eq__(item)
    
    def __str__(self):
        """
        Get a string representation of the stack.
        
        `AtomArray` strings eparated by blank lines
        and a line indicating the index.
        """
        string = ""
        for i, array in enumerate(self):
            string += "Model: " + str(i) + "\n"
            string += str(array) + "\n" + "\n"
        return string


def stack(arrays):
    """
    Create an `AtomArrayStack` from a list of `AtomArray`.
    
    All atom arrays must have equal annotation arrays.
    
    Parameters
    ----------
    arrays : Iterable object of AtomArray
        The atom arrays to be combined in a stack.
    
    Returns
    -------
    stack : AtomArrayStack
        The stacked atom arrays.
    """
    for array in arrays:
        # Check if all arrays share equal annotations
        if not super(AtomArray, array).__eq__(arrays[0]):
            raise ValueError("The arrays atom annotations"
                             "do not fit to each other") 
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
    """
    Create an `AtomArray` from a `Bio.PDB.Model.Model`.
    
    Parameters
    ----------
    model : Model
        All atoms of the model are included in the atom array.
    insertion_code: string, optional
        Since each atom may only occur once in an `AtomArray`, you have to
        choose which insertion code to use. By default no insertion code is
        expected.
    
    Returns
    -------
    array : AtomArray
        The resulting atom array.
        
    See Also
    --------
    to_model
    
    Notes
    -----
    Currently this does not support alternative atom locations. If you want to
    have give an atom an alternative location, you have to do that manually.
    """
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
    """
    Create a `Bio.PDB.Model.Model` from an `AtomArray`.
    
    This does the reverse to `to_array()`.
    
    Parameters
    ----------
    array : AtomArray
        The atom array to be converted to a model.
    
    Returns
    -------
    model : Model
        The resulting model.
        
    See Also
    --------
    to_array
    """
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
        # if chain does not exist create chain
        # and add it to super entity (model)
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
    """
    Calculate the number of atoms in a model.
    """
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
    """
    Get the insertion code of a residue.
    """
    return residue.id[2].strip()


def position(item):
    """
    Get the atom position of the given array.
    
    This may be directly and `AtomArray` or `AtomArrayStack` or alternatively
    an (n x 3) or (m x n x 3) `ndarray` containing the coordinates.
    
    Parameters
    ----------
    item : `AtomArray` or `AtomArrayStack` or ndarray
        Takes the pos attribute, if `item` is `AtomArray` or `AtomArrayStack`,
        or takes directly a ndarray.
    
    Returns
    -------
    pos : ndarray
        Atom coordinates.
    """

    if type(item) in (Atom, AtomArray, AtomArrayStack):
        return item.pos
    else:
        return np.array(item)