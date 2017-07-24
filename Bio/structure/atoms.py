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

The following annotation categories are mandatory:

=========  ===========  ===================   ================================
Category   Type         Examples              Description
=========  ===========  ===================   ================================
chain_id   string (U3)  'A','S','AB', ...     Polypeptide chain
res_id     int          1,2,3, ...            Sequence position of residue
res_name   string (U3)  'GLY','ALA', ...      Residue name
hetero     bool         True, False           Identifier for non AA residues
atom_name  string (U6)  'CA','N', ...         Atom name
element    string (U2)  'C','O','N', ...      Chemical Element
=========  ===========  ===================   ================================

These annotation categories correspond to `Entity` attributes in `Bio.PDB`. For
all `Atom`, `AtomArray` and `AtomArrayStack` objects these annotations must be
set, otherwise some functions will not work or errors will occur. Additionally
to these annotations, an arbitrary amount of annotation categories can be added
(Use `add_annotation()` for this). The annotation arrays can be accessed either
via the corresponding dictionary (e.g. ``array._annot["res_id"]``) or directly
(e.g. ``array.res_id``).

For each type, the attributes can be accessed directly. Both `AtomArray` and
`AtomArrayStack` support `numpy` style indexing, the index is propagated to
each attribute. If a single integer is used as index, an object with one
dimension less is returned
(`AtomArrayStack` -> `AtomArray`, `AtomArray` -> `Atom`).
Do not expect a deep copy, when sclicing an `AtomArray` or `AtomArrayStack`.
The attributes of the sliced object may still point to the original `ndarray`. 
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
        self._annot = {}
        self.add_annotation("chain_id")
        self.add_annotation("res_id")
        self.add_annotation("res_name")
        self.add_annotation("hetero")
        self.add_annotation("atom_name")
        self.add_annotation("element")
        if length == None:
            return
        self.chain_id = np.zeros(length, dtype="U3")
        self.res_id = np.zeros(length, dtype=int)
        self.res_name = np.zeros(length, dtype="U3")
        self.hetero = np.zeros(length, dtype=bool)
        self.atom_name = np.zeros(length, dtype="U6")
        self.element = np.zeros(length, dtype="U2")
        
    def add_annotation(self, annotation):
        """
        Add an annotation category, if not already existing.
        
        Parameters
        ----------
        annotation : string
            The annotation category to be added.
        """
        if annotation not in self._annot:
            self._annot[str(annotation)] = None
            
    def get_annotation(self, annotation):
        """
        Return an annotation array.
        
        Parameters
        ----------
        annotation : string
            The annotation category to be returned.
            
        Returns
        -------
        array : ndarray
            The annotation array.
        """
        if annotation not in self._annot:
            raise ValueError("Annotation category '" + annotation + "' is not existing")
        return self._annot[annotation]
    
    def set_annotation(self, annotation, array):
        """
        Set an annotation array. if the annotation category does not exist yet, the category is
        created.
        
        Parameters
        ----------
        annotation : string
            The annotation category to be set.
        array : string
            The new value of the annotation category.
        """
        self._annot[annotation] = array
        
    def get_annotation_categories(self):
        """
        Return a list containing all annotation categories.
            
        Returns
        -------
        categories : list
            The list containing the names of each annotation category.
        """
        return list(self._annot.keys())
    
    def __getattr__(self, attr):
        """
        If the attribute is an annotation, the annotation is returned from
        the dictionary.
        """
        if attr in self._annot:
            return self._annot[attr]
        else:
            raise AttributeError("'" + type(self).__name__ +
                                 "' object has no attribute '" + attr + "'")
        
    def __setattr__(self, attr, value):
        """
        If the attribute is an annotation, the `value` is saved to the
        annotation in the dictionary.
        """
        # First condition is required, since call of the second would result in
        # indefinite calls of __getattr__
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr in self._annot:
            self._annot[attr] = value
        else:
            super().__setattr__(attr, value)

        
    def check_integrity(self):
        """
        Check, if all annotation ndarrays have appropriate shapes.
        
        Returns
        -------
        integrity : bool
            True, if the attribute shapes are consistent.
        """
        for annotation in self._annot.values():
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
        if self._annot.keys() != item._annot.keys():
            return False
        for name in self._annot:
            if not np.array_equal(self._annot[name], item._annot[name]):
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
            return [((self.atom_name == "N") |
                   (self.atom_name == "CA") |
                   (self.atom_name == "C")) &
                   (self.hetero == False)]
        elif filter == "hetero":
            return self.hetero == True
        elif "E=" in filter:
            element = filter[-1]
            return np.array(
                [name.strip()[0] == element for name in self.atom_name])
        else:
            filter = filter.strip()
            return np.array(
                [name.strip() == filter for name in self.atom_name])
            
    def annotation_length(self):
        """
        Get the length of the annotation arrays.
        
        For AtomArray it is the same as ``len(array)``.
        
        Returns
        -------
        length : int
            Length of the annotation arrays.
        """
        return len(self.chain_id)
    
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
    
    Attributes
    ----------
    annot : dict
        The dictionary containing all annotations.
    coord : ndarray(dtype=float)
        ndarray containing the x, y and z coordinate of the atom.
    """
    
    def __init__(self, coord, **kwargs):
        """
        Create an `Atom`.
        
        Parameters
        ----------
        See class attributes
        """
        self._annot = {}
        if "kwargs" in kwargs:
            # kwargs are given directly as dictionary
            kwargs = kwargs["kwargs"]
        for name, annotation in kwargs.items():
            self._annot[name] = annotation
        coord = np.array(coord, dtype=float)
        # Check if coord contains x,y and z coordinates
        if coord.shape != (3,):
            raise ValueError("Position must be ndarray with shape (3,)")
        self.coord = coord
        
    def __getattr__(self, attr):
        if attr in self._annot:
            return self._annot[attr]
        else:
            raise AttributeError("'" + type(self).__name__ +
                                 "' object has no attribute '" + attr + "'")
        
    def __setattr__(self, attr, value):
        # First condition is required, since call of the second would result in
        # indefinite calls of __getattr__
        if attr == "_annot":
            super().__setattr__(attr, value)
        elif attr in self._annot:
            self._annot[attr] = value
        else:
            super().__setattr__(attr, value)
    
    def __str__(self):
        """
        String representation of the atom.
        """
        string = ""
        for value in self._annot.values():
            string += str(value) + "\t"
        return string + str(self.coord)

    
class AtomArray(_AtomAnnotationList):
    """
    An array representation of a structure consisting of multiple atoms.
    
    All attributes correspond to `Entity` attributes in `Bio.PDB`.
    
    Attributes
    ----------
    annot : dict
        The dictionary containing all annotation arrays.
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
        for name in self._annot:
            new_array._annot[name] = np.copy(self._annot[name])
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
        for name, annotation in self._annot.items():
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
        if isinstance(index, int):
            return self.get_atom(index)
        elif isinstance(index, tuple):
            if len(index) == 2 and index[0] is Ellipsis:
                # If first index is "...", just ignore the first index
                return self.__getitem__(index[1])
            else:
                raise IndexError("AtomArray cannot take multidimensional"
                                 "indices")
        else:
            new_array = AtomArray()
            for annotation in self._annot:
                new_array._annot[annotation] = (self._annot[annotation]
                                                  .__getitem__(index))
            new_array.coord = self.coord.__getitem__(index)
            return new_array
        
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
            for name in self._annot:
                self._annot[name] = atom._annot[name]
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
            for name in self._annot:
                self._annot[name] = np.delete(self._annot[name], index, axis=0)
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
    annot : dict
        The dictionary containing all annotation arrays.
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
        for name in self._annot:
            new_stack._annot[name] = np.copy(self._annot[name])
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
        for name in self._annot:
            array._annot[name] = self._annot[name]
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
            otherwise an `AtomArrayStack` with reduced depth and length
            is returned. In case the index is a tuple(int, int) an `Atom`
            instance is returned.  
        """
        if isinstance(index, int):
            return self.get_array(index)
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("AtomArrayStack can take an index with more "
                                 "than two dimensions")
            if type(index[0]) == int:
                if type(index[1]) == int:
                    array = self.get_array(index[0])
                    return array.get_atom(index[1])
                else:
                    array = self.get_array(index[0])
                    return array.__getitem__(index[1])
            else:
                new_stack = AtomArrayStack()
                for name in self._annot:
                    new_stack._annot[name] = (self._annot[name]
                                                .__getitem__(index[1]))
                if index[0] is Ellipsis:
                    new_stack.coord = self.coord[:,index[1]]
                else:
                    new_stack.coord = self.coord.__getitem__(index)
                return new_stack
        else:
            new_stack = AtomArrayStack()
            for name in self._annot:
                new_stack._annot[name] = (self._annot[name])
            new_stack.coord = self.coord.__getitem__(index)
            return new_stack
            
    
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
            string += "Model " + str(i+1) + "\n"
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
    names = sorted(atoms[0]._annot.keys())
    for atom in atoms:
        if sorted(atom._annot.keys()) != names:
            raise ValueError("The atoms do not share the"
                             "same annotation categories")
    # Add all atoms to AtomArray
    array = AtomArray(length=len(atoms))
    for i in range(len(atoms)):
        for name in names:
            array._annot[name] = atoms._annot[name]
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
    for name, annotation in arrays[0]._annot.items():
        array_stack._annot[name] = annotation
    coord_list = [array.coord for array in arrays] 
    array_stack.coord = np.stack(coord_list, axis=0)
    return array_stack


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