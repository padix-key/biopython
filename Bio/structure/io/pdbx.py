# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import shlex
import numpy as np

class PDBxFile(object):
    
    def __init__(self):
        pass
    
    
    def read(self, file_name):
        with open(file_name, "r") as f:
            str_data = f.read()
        self._lines = str_data.split("\n")
        # This dictionary saves the PDBx category names,
        # together with its line position in the file
        self._data_blocks = {}
        
        current_data_block = ""
        current_category = None
        start = -1
        stop = -1
        is_loop = False
        has_multiline_values = False
        for i, line in enumerate(self._lines):
            # Ignore empty and comment lines
            if not _is_empty(line):
                data_block_name = _data_block_name(line)
                if data_block_name is not None:
                    data_block = self._add_data_block(data_block_name)
                    # If new data block begins, reset category data
                    current_category = None
                    start = -1
                    stop = -1
                    is_loop = False
                    has_multiline_values = False
                
                is_loop_in_line = _is_loop_start(line)
                category_in_line = get_category_name(line)
                if is_loop_in_line or (category_in_line != current_category
                                       and category_in_line is not None):
                    # Start of a new category
                    # Add an entry into the dictionary with the old category
                    stop = i
                    self._add_category(data_block, current_category, start,
                                       stop, is_loop, has_multiline_values)
                    # Track the new category
                    if is_loop_in_line:
                        # In case of lines with "loop_" the category is in the
                        # next line
                        category_in_line = get_category_name(self._lines[i+1])
                    is_loop = is_loop_in_line
                    current_category = category_in_line
                    start = i
                    has_multiline_values = False
                
                multiline = _is_multi(line, is_loop)
                if multiline:
                    has_multiline_values = True
        # Add the entry for the final category
        # Since at the end of the file the end of the category
        # is not determined by the start of a new one,
        # this needs to be handled separately
        stop = len(self._lines)
        self._add_category(data_block, current_category, start,
                           stop, is_loop, has_multiline_values)
    
    
    def get_block_names(self):
        return list(self._data_blocks.keys())
    
    
    def get_category(self, block, category):
        category_info = self._data_blocks[block][category]
        start = category_info[0]
        stop = category_info[1]
        is_loop = category_info[2]
        is_multilined = category_info[3]
        
        if is_multilined:
            pre_lines = [line.strip() for line in self._lines[start:stop]
                         if not _is_empty(line) and not _is_loop_start(line)]
            lines = (len(pre_lines)) * [None]
            k = 0
            i = 0
            while i < len(pre_lines):
                if pre_lines[i][0] == ";":
                    lines[k-1] += " '" + pre_lines[i][1:]
                    j = i+1
                    while pre_lines[j] != ";":
                        lines[k-1] += " " + pre_lines[j]
                    lines[k-1] += "'"
                    i = j+1
                elif not is_loop and pre_lines[i][0] in ["'",'"']:
                    lines[k-1] += " " + pre_lines[i]
                    i += 1
                else:    
                    lines[k] = pre_lines[i]
                    i += 1
                    k += 1
            lines = [line for line in lines if line is not None]
            
        else:
            lines = [line.strip() for line in self._lines[start:stop]
                     if not _is_empty(line) and not _is_loop_start(line)]
        
        if is_loop:
            if is_multilined:
                category_dict = self._process_looped_multilined(lines)
            else:
                category_dict = self._process_looped(lines)
        else:
            category_dict = self._process_singlevalued(lines)
        
        return category_dict
            
    
    def set_category(self, block, category, data):
        raise NotImplementedError()
    
    
    def write(self, file_name):
        raise NotImplementedError()
    
    
    def copy(self):
        PDBxFile = PDBxFile()
        PDBxFile._lines = copy.deepcopy(self._lines)
        PDBxFile._data_blocks = copy.deepcopy(self._data_blocks)
    
    
    def _add_data_block(self, block_name):
        self._data_blocks[block_name] = {}
        return self._data_blocks[block_name]
    
    
    def _add_category(self, block, category_name,
                      start, stop, is_loop, is_multilined):
        # Before the first category starts,
        # the current_category is None
        # This is checked before adding an entry
        if category_name is not None:
            block[category_name] = (start, stop, is_loop, is_multilined)
    
            
    def _process_singlevalued(self, lines):
        category_dict = {}
        for line in lines:
            parts = shlex.split(line)
            key = parts[0].split(".")[1]
            value = parts[1]
            category_dict[key] = value
        return category_dict
    
    
    def _process_looped(self, lines):
        category_dict = {}
        keys = []
        i = 0
        for line in lines:
            in_key_lines = (line[0] == "_")
            if in_key_lines:
                key = line.split(".")[1]
                keys.append(key)
                category_dict[key] = np.zeros(len(lines),
                                              dtype="U64")
                keys_length = len(keys)
            else:
                values = shlex.split(line)
                j = 0
                while j < keys_length:
                    category_dict[keys[j]][i] = values[j]
                    j += 1
                i += 1
        for key in category_dict.keys():
            category_dict[key] = category_dict[key][:i]
        return category_dict
    
    
    def _process_looped_multilined(self, lines):
        raise NotImplementedError()
        category_dict = {}
        return category_dict

def _is_empty(line):
    return len(line) == 0 or line[0] == "#"


def _data_block_name(line):
    if line.startswith("data_"):
        return line[5:]
    else:
        return None

def _is_loop_start(line):
    return line.startswith("loop_")


def _is_multi(line, is_loop):
    if is_loop:
        return line[0] == ";"
    else:
        return line[0] in [";","'",'"']


def get_category_name(line):
    if line[0] != "_":
        return None
    else:
        return line[1:line.find(".")]
    