# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from . import Atom, AtomArray, AtomArrayStack

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