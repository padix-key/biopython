# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module contains the main types of the `Structure` subpackage: `Atom`,
`AtomArray` and `AtomArrayStack`.

In this context an atom is described by two kinds of attributes: the
coordinates and the annotations. The annotations include information about
polypetide chain id, residue id, residue name, hetero atom information and atom
name. The coordinates are a `numpy` float ndarray of length 3, containing the
x, y and z coordinates.

An `Atom` contains data for a single atom, it stores the annotations as scalar
values and the coordinates as length 3 ndarray.
An `AtomArray` stores data for an entire model containing *n* atoms.
Therefore the annotations are represented as ndarrays of length *n*, so called
annotation arrays. The coordinates are a (n x 3) ndarray.
`AtomArrayStack` stores data for *m* models. Each `AtomArray` in
the `AtomArrayStack` has the same annotation arrays, but may differ in atom
coordinates. Therefore the annotation arrays are represented as ndarrays of
length *n*, while the coordinates are a (m x n x 3) ndarray.
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
    
    def __init__(self, length=None):
        """
        Create the annotation arrays
        """
        self.annot = {}
        self.add_annotation("chain_id")
        self.add_annotation("res_id")
        self.add_annotation("res_name")
        self.add_annotation("atom_name")
        self.add_annotation("hetero")
        self.add_annotation("element")
        if length == None:
            return
        # string size based on reserved columns in *.pdb files
        self.chain_id = np.zeros(length, dtype="U1")
        self.res_id = np.zeros(length, dtype=int)
        self.res_name = np.zeros(length, dtype="U3")
        self.atom_name = np.zeros(length, dtype="U4")
        self.hetero = np.zeros(length, dtype="U5")
        self.element = np.zeros(length, dtype="U1")
        
    def add_annotation(self, annotation):
        if annotation not in self.annot:
            self.annot[annotation] = None
    
    def __getattr__(self, attr):
        if attr in self.annot:
            return self.annot[attr]
        else:
            raise AttributeError("'" + "attr" + "' is not a valid atom annotation")
        
    def __setattr__(self, attr, value):
        # First condition is required, since call of the second would result in
        # indefinite calls of __getattr__
        if attr == "annot":
            super().__setattr__(attr, value)
        elif attr in self.annot:
            self.annot[attr] = value
        else:
            super().__setattr__(attr, value)
        
        
        
    def seq_length(self, chain_id="all"):
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
        integrity : bool
            True, if the attribute shapes are consistent.
        """
        for annotation in self.annot.values():
            if annotation.shape != (len(self),):
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
        equality : bool
            True, if the aannotation arrays are equal.
        """
        if not isinstance(item, _AtomAnnotationList):
            return False
        if self.annot.keys() != item.annot.keys():
            return False
        for name in self.annot:
            if not np.array_equal(self.annot[name], item.annot[name]):
                return False
        return True
    
    def filter_atoms(self, filter):
        """
        Create a filter for defined atom names.
        
        This method has a similar result to ``object.atom_name == filter``.
        The difference is that this method trims whitespaces from the compared
        values. Furthermore this method can take special filter values, to
        filter for groups of atom names or element names:
        
            - "all": No filtering is applied
            - "backbone": Filters all protein backbone atoms ("N","CA","C")
            - "hetero": Filters all atoms from hetero residues
            - "E=X" Filters all atom of element *X*
            
        Since this method involves whitespace stripping, the performance is
        lower compared to ``object.atom_name == filter``.
        
        Parameters
        ----------
        filter : string
            The object to compare the annotation arrays with.
        
        Returns
        -------
        filter_array : 1-D ndarray(dtype=bool)
            Filter for the given atom names, used for fancy indexing. Use atom
            names or the special filters mentioned above.
        """
        if filter == "all":
            return np.ones(len(self), dtype=bool)
        elif filter == "backbone":
            return [((self.atom_name == " N  ") |
                   (self.atom_name == " CA ") |
                   (self.atom_name == " C  ")) &
                   (self.hetero == " ")]
        elif filter == "hetero":
            return self.hetero != " "
        elif "E=" in filter:
            element = filter[-1]
            return np.array(
                [name.strip()[0] == element for name in self.atom_name])
        else:
            filter = filter.strip()
            return np.array(
                [name.strip() == filter for name in self.atom_name])
    
    def __eq__(self, item):
        """
        See Also
        --------
        equal_annotations
        """
        return self.equal_annotations(item)
    
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
        Integer value identifying the sequence position of the residue
    res_name : string {'GLY','ALA',...}
        Three character string representing the residue name
    atom_name : string {' CA ',' N  ',...}
        Four character string representing the atom name.
        Pay attention to the whitespaces
    element: string {'C','O','N',...}
        A single character representing the element.
    hetero : string {' ','W','H_GLC',...}
        Up to 5 character string, indicating in which hetero residue the
        atom is in. If the residue is a standard amino acid the value is `' '`.
    """
    
    def __init__(self, coord, **kwargs):
        """
        Create an `Atom`.
        
        Parameters
        ----------
        See class attributes
        """
        self.annot = {}
        if "kwargs" in kwargs:
            # kwargs are given directly as dictionary
            kwargs = kwargs["kwargs"]
        for name, annotation in kwargs.items():
            annot[name] = annotation
        coord = np.array(coord, dtype=float)
        # Check if coord contains x,y and z coordinates
        if coord.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.coord = coord
        
    def __getattr__(self, attr):
        if attr in self.annot:
            return self.annot[attr]
        else:
            raise AttributeError("'" + "attr" + "' is not a valid atom annotation")
        
    def __setattr__(self, attr, value):
        # First condition is required, since call of the second would result in
        # indefinite calls of __getattr__
        if attr == "annot":
            super().__setattr__(attr, value)
        elif attr in self.annot:
            self.annot[attr] = value
        else:
            super().__setattr__(attr, value)
    
    def __str__(self):
        """
        String representation of the atom.
        """
        string = ""
        for value in self.annot.values():
            string += str(value) + "\t"
        return string + "\t" + str(self.coord)

    
class AtomArray(_AtomAnnotationList):
    """
    An array representation of a structure consisting of multiple atoms.
    
    All attributes correspond to `Entity` attributes in `Bio.PDB`.
    
    Attributes
    ----------
    chain_id : ndarray(dtype="U1") {'A','B',...}
        A single character representing the polypeptide chain
    res_id : ndarray(dtype=int)
        Integer value identifying the sequence position of the residue
    res_name : ndarray(dtype="U3") {'GLY','ALA',...}
        Three character string representing the residue name
    atom_name : ndarray(dtype="U4") {' CA ',' N  ',...}
        Four character string representing the atom name.
        Pay attention to the whitespaces
    element: string {'C','O','N',...}
        A single character representing the element.
    hetero : ndarray(dtype="U5") {' ','W','H_GLC',...}
        Up to 5 character string, indicating in which hetero residue the
        atom is in. If the residue is a standard amino acid the value is `' '`.
    coord : ndarray(dtype=float)
        (n x 3) ndarray containing the x, y and z coordinate of the atoms.
    """
    
    def __init__(self, length=None):
        """
        Create an `AtomArray`.
        
        Parameters
        ----------
        length : int, optional
            If length is given, the attribute arrays will be created with
            zeros.
        """
        super().__init__(length)
        if length == None:
            return
        self.coord = np.zeros((length, 3), dtype=float)
        
    def copy(self):
        """
        Create a new `AtomArray` instance with all attribute arrays copied.
        
        Returns
        -------
        new_array : AtomArray
            A deep copy of this array.
        """
        new_array = AtomArray()
        for name in self.annot:
            new_array.annot[name] = np.copy(self.annot[name])
        new_array.coord = np.copy(self.coord)
        return new_array
        
    def check_integrity(self):
        """
        Check, if all attribute arrays have appropriate shapes.
        
        Returns
        -------
        integrity : bool
            True, if the attribute shapes are consistent.
        """
        if not super().check_integrity():
            return False
        if self.coord.shape != (len(self), 3):
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
        kwargs = {}
        for name, annotation in self.annot:
            kwargs[name] = annotation[index]
        return Atom(coord = self.coord[index], kwargs=kwargs)
    
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
            If `index` is an integer an `Atom` instance,
            otherwise an `AtomArray` with reduced length is returned.
        """
        try:
            if isinstance(index, int):
                return self.get_atom(index)
            else:
                new_array = AtomArray()
                for annotation in self.annot:
                    new_array.annot[annotation] = self.annot[annotation].__getitem__(index)
                new_array.coord = self.coord.__getitem__(index)
                return new_array
        except:
            raise IndexError("Invalid index") from None
        
    def __setitem__(self, index, atom):
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
            for name in self.annot:
                self.annot[name] = atom.annot[name]
            self.coord[index] = atom.coord
        else:
            raise IndexError("Index must be integer")
        
    def __delitem__(self, index):
        """
        Deletes the atom at the specified array position.
        
        Parameters
        ----------
        index : int
            The position where the atom should be deleted.
        """
        if isinstance(index, int):
            for name in self.annot:
                self.annot[name] = np.delete(self.annot[name], index, axis=0)
            self.coord = np.delete(self.coord, index, axis=0)
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
        # length is determined by length of coord attribute
        return self.coord.shape[0]
    
    def __eq__(self, item):
        """
        Check if the array equals another `AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : bool
            True, if `item` is an `AtomArray`
            and all its attribute arrays equals the ones of this object.
        """
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArray):
            return False
        if not np.array_equal(self.coord, item.coord):
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
    1-D, while the coordinate array is 3-D (m x n x 3).
    
    All attributes correspond to `Entity` attributes in `Bio.PDB`.
    
    Attributes
    ----------
    chain_id : ndarray(dtype="U1") {'A','B',...}
        A single character representing the polypeptide chain
    res_id : ndarray(dtype=int)
        Integer value identifying the sequence position of the residue
    res_name : ndarray(dtype="U3") {'GLY','ALA',...}
        Three character string representing the residue name
    atom_name : ndarray(dtype="U4") {' CA ',' N  ',...}
        Four character string representing the atom name.
        Pay attention to the whitespaces
    element: string {'C','O','N',...}
        A single character representing the element.
    hetero : ndarray(dtype="U5") {' ','W','H_GLC',...}
        Up to 5 character string, indicating in which hetero residue the
        atom is in. If the residue is a standard amino acid the value is `' '`.
    coord : ndarray(dtype=float)
        (m x n x 3) ndarray containing the x, y and z coordinate of the atoms.
    """
    
    def __init__(self, depth=None, length=None):
        """
        Create an `AtomArrayStack`.
        
        Parameters
        ----------
        depth, length : int, optional
            If length and depth is given, the attribute arrays will be created
            with zeros. `depth` corresponds to the first dimension, `length`
            to the second.
        """
        super().__init__(length)
        if depth == None or length == None:
            return
        self.coord = np.zeros((depth, length, 3), dtype=float)
        
    def copy(self):
        """
        Create a new `AtomArrayStack` instance
        with all attribute arrays copied.
        
        Returns
        -------
        new_stack: AtomArrayStack
            A deep copy of this stack.
        """
        new_stack = AtomArrayStack()
        for name in self.annot:
            new_stack.annot[name] = np.copy(self.annot[name])
        new_stack.coord = np.copy(self.coord)
        return new_stack
    
    def check_integrity(self):
        """
        Check, if all attribute arrays have appropriate shapes.
        
        Returns
        -------
        integrity : bool
            True, if the attribute shapes are consistent.
        """
        if not super().check_integrity():
            return False
        if self.coord.shape != (len(self), super().__len__(), 3):
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
        for name in self.annot:
            array.annot[name] = self.annot[name]
        array.coord = self.coord[index]
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
            If `index` is an integer an `AtomArray` instance,
            otherwise an `AtomArrayStack` with reduced depth abd length
            is returned. In case the index is a tuple(int, int) an `Atom`
            instance is returned.  
        """
        try:
            if isinstance(index, int):
                return self.get_array(index)
            elif isinstance(index, tuple):
                if type(index[0]) == int and type(index[1]) == int:
                    array = self.get_array(index[0])
                    return array.get_atom(index[1])
                else:
                    new_stack = AtomArrayStack()
                    for name in self.annot:
                        new_stack.annot[name] = self.annot[name].__getitem__(index[1:])
                    new_stack.coord = self.coord.__getitem__(index)
                    return new_stack
            else:
                new_stack = AtomArrayStack()
                for name in self.annot:
                    new_stack.annot[name] = self.annot[name].__getitem__(index)
                    new_stack.coord = self.coord.__getitem__(index)
                return new_stack
        except:
            raise IndexError("Invalid index")
            
    
    def __setitem__(self, index, array):
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
            self.coord[index] = array.coord
        else:
            raise IndexError("Index must be integer")
        
    def __delitem__(self, index):
        """
        Deletes the atom array at the specified stack position.
        
        Parameters
        ----------
        index : int
            The position where the atom array should be deleted.
        """
        if isinstance(index, int):
            self.coord = np.delete(self.coord, index, axis=0)
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
        # length is determined by length of coord attribute
        return self.coord.shape[0]
    
    def __eq__(self, item):
        """
        Check if the array equals another `AtomArray`
        
        Parameters
        ----------
        item : object
            Object to campare the array with.
        
        Returns
        -------
        equal : bool
            True, if `item` is an `AtomArray`
            and all its attribute arrays equals the ones of this object.
        """
        if not super().__eq__(item):
            return False
        if not isinstance(item, AtomArrayStack):
            return False
        if not np.array_equal(self.coord, item.coord):
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

def array(atoms):
    """
    Create an `AtomArray` from a list of `Atom`.
    
    Parameters
    ----------
    atoms : array_like(Atom)
        The atoms to be combined in an array.
    
    Returns
    -------
    array : AtomArray
        The listed atoms as array.
    """
    # Check if all atoms have the same annotation names
    # Equality check requires sorting
    names = sorted(atoms[0].annot.keys())
    for atom in atoms:
        if sorted(atom.annot.keys()) != names:
            raise ValueError("The atoms do not share the"
                             "same annotation categories")
    # Add all atoms to AtomArray
    array = AtomArray(length=len(atoms))
    for i in range(len(atoms)):
        for name in names:
            array.annot[name] = atoms.annot[name]
        array.coord[i] = atoms[i].coord
    return array

def stack(arrays):
    """
    Create an `AtomArrayStack` from a list of `AtomArray`.
    
    All atom arrays must have equal annotation arrays.
    
    Parameters
    ----------
    arrays : array_like(AtomArray)
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
    for name, annotation in arrays[0].annot.items():
        array_stack.annot[name] = annotation
    coord_list = [array.coord for array in arrays] 
    array_stack.coord = np.stack(coord_list, axis=0)
    return array_stack

def to_array(model, insertion_code=""):
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
                    arr.atom_name[i] = atom.get_fullname()
                    arr.element[i] = atom.get_name().strip()[0]
                    arr.coord[i] = atom.get_coord()
                    i += 1
    return arr


def to_model(array, id=0):
    """
    Create a `Bio.PDB.Model.Model` from an `AtomArray`.
    
    This does the reverse to `to_array()`.
    
    Parameters
    ----------
    array : AtomArray
        The atom array to be converted to a model.
    id : int
        ID of the model.
    
    Returns
    -------
    model : Model
        The resulting model.
        
    See Also
    --------
    to_array
    """
    model = Bio.PDB.Model.Model(id, id+1)
    # Iterate through all atoms
    for i in range(len(array)):
        # Extract annotations and coordinates of every atom
        chain_id = array.chain_id[i]
        hetero = array.hetero[i]
        res_id = array.res_id[i]
        res_name = array.res_name[i]
        atom_name = array.atom_name[i]
        element = array.element[i]
        coord = array.coord[i]
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
            atom_curr = Bio.PDB.Atom.Atom(atom_name, coord, 0, 1, " ",
                                          atom_name, i+1, element)
            res_curr.add(atom_curr)
    return model


def to_array_stack(structure, insertion_code=""):
    """
    Create an `AtomArrayStack` from a `Bio.PDB.Structure.Structure`.
    
    Parameters
    ----------
    structure : Structure
        All atoms of the structure are included in the atom array stack,
        while each model yields one atom array in the stack.
    insertion_code: string, optional
        Since each atom may only occur once in an `AtomArray`, you have to
        choose which insertion code to use. By default no insertion code is
        expected.
    
    Returns
    -------
    stack : AtomArrayStack
        The resulting atom stack.
        
    See Also
    --------
    to_structure
    
    Notes
    -----
    Currently this does not support alternative atom locations. If you want to
    have give an atom an alternative location, you have to do that manually.
    """
    return stack(
        [to_array(model, insertion_code) for model in structure])


def to_structure(stack, id=""):
    """
    Create a `Bio.PDB.Structure.Structure` from an `AtomArrayStack`.
    
    This does the reverse to `to_array_stack()`.
    
    Parameters
    ----------
    stack : AtomArrayStack
        The atom array to be converted to a model.
    id : string
        ID of the structure.
    
    Returns
    -------
    structure : Structure
        The resulting structure.
        
    See Also
    --------
    to_array_stack
    """
    structure = Bio.PDB.Structure.Structure(id)
    for i, array in enumerate(stack):
        model = to_model(array, i)
        structure.add(model)
    return structure


def _get_model_size(model, insertion_code=""):
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


def _get_insertion_code(residue):
    """
    Get the insertion code of a residue.
    """
    return residue.id[2].strip()


def coord(item):
    """
    Get the atom coordinates of the given array.
    
    This may be directly and `AtomArray` or `AtomArrayStack` or alternatively
    an (n x 3) or (m x n x 3) `ndarray` containing the coordinates.
    
    Parameters
    ----------
    item : `AtomArray` or `AtomArrayStack` or ndarray
        Takes the coord attribute, if `item` is `AtomArray` or
        `AtomArrayStack`, or takes directly a ndarray.
    
    Returns
    -------
    coord : ndarray
        Atom coordinates.
    """

    if type(item) in (Atom, AtomArray, AtomArrayStack):
        return item.coord
    else:
        return np.array(item)