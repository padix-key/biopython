# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from ... import BadStructureError
from ... import Atom, AtomArray, AtomArrayStack    


def get_structure(pdbx_file, data_block=None, insertion_code=None,
                  altloc=None, model=None, extra_fields=None):
    if data_block is None:
        data_block = pdbx_file.get_block_names()[0]
    atom_site_dict = pdbx_file.get_category(data_block, "atom_site")
    models = atom_site_dict["pdbx_PDB_model_num"]
    if model is None:
        stack = AtomArrayStack()
        # For a stack, the annotation are derived from the first model
        model_dict = _get_model_dict(atom_site_dict, 1)
        _fill_annotations(stack, model_dict, extra_fields)
        model_count = int(models[-1])
        model_length = len(stack.seg_id)
        # Check if each model has the same amount of atoms
        # If not, raise exception
        if model_length * model_count != len(models):
            raise BadStructureError("The models in the file have unequal"
                            "amount of atoms, give an explicit model instead")
        stack.coord = np.zeros((model_count, model_length, 3), dtype=float)
        stack.coord[:,:,0] = atom_site_dict["Cartn_x"].reshape((model_count,
                                                                model_length))
        stack.coord[:,:,1] = atom_site_dict["Cartn_y"].reshape((model_count,
                                                                model_length))
        stack.coord[:,:,2] = atom_site_dict["Cartn_z"].reshape((model_count,
                                                                model_length))
        stack = _filter_inscode_altloc(stack, model_dict,
                                       insertion_code, altloc)
        return stack
    else:
        array = AtomArray()
        model_dict = _get_model_dict(atom_site_dict, model)
        _fill_annotations(array, model_dict, extra_fields)
        model_length = len(array.seg_id)
        model_filter = (models == str(model))
        array.coord = np.zeros((model_length, 3), dtype=float)
        array.coord[:,0]= atom_site_dict["Cartn_x"][model_filter].astype(float)
        array.coord[:,1]= atom_site_dict["Cartn_y"][model_filter].astype(float)
        array.coord[:,2]= atom_site_dict["Cartn_z"][model_filter].astype(float)
        array = _filter_inscode_altloc(array, model_dict,
                                       insertion_code, altloc)
        return array
        

def _fill_annotations(array, model_dict, extra_fields):
    array.annot["seg_id"] = model_dict["label_entity_id"].astype(int)
    array.annot["res_id"] = np.array([-1 if e in [".","?"] else int(e)
                                      for e in model_dict["label_seq_id"]])
    array.annot["res_name"] = model_dict["label_comp_id"].astype("U3")
    array.annot["hetero"] = (model_dict["group_PDB"] == "HETATM")
    array.annot["atom_name"] = model_dict["label_atom_id"].astype("U6")
    array.annot["element"] = model_dict["type_symbol"].astype("U2")
    if extra_fields is not None:
        for field in extra_fields:
            field_name = field[0]
            annot_name = field[1]
            array.annot[annot_name] = model_dict[field_name]


def _filter_inscode_altloc(array, model_dict, inscode, altloc):
    inscode_array = model_dict["pdbx_PDB_ins_code"]
    altloc_array = model_dict["label_alt_id"]
    # Default: Filter all atoms with insertion code ".", "?" or "A"
    inscode_filter = np.in1d(inscode_array, [".","?","A"],
                             assume_unique=True)
    # Now correct filter for every given insertion code
    if inscode is not None:
        for code in inscode:
            residue = code[0]
            insertion = code[1]
            residue_filter = (array.res_id == residue)
            # Resetet filter for given res_id
            inscode_filter &= ~residue_filter
            # Choose atoms of res_id with insertion code
            inscode_filter |= residue_filter & (inscode_array == insertion)
    # Same with altlocs
    altloc_filter = np.in1d(altloc_array, [".","?","A"],
                            assume_unique=True)
    if altloc is not None:
        for loc in altloc:
            residue = loc[0]
            altloc = loc[1]
            residue_filter = (array.res_id == residue)
            altloc_filter &= ~residue_filter
            altloc_filter |= residue_filter & (altloc_array == altloc)
    # Apply combined filters
    return array[inscode_filter & altloc_filter]
    


def _get_model_dict(atom_site_dict, model):
    model_dict = {}
    models = atom_site_dict["pdbx_PDB_model_num"]
    for key in atom_site_dict.keys():
        model_dict[key] = atom_site_dict[key][models == str(model)]
    return model_dict


def set_structure(pdbx_file, array):
    pass


def set_coord(pdbx_file, coord):
    pass