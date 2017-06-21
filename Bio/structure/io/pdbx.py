# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import shlex

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
                    self._add_category(data_block, current_category,
                                       start, stop, is_loop)
                    # Track the new category
                    if is_loop_in_line:
                        # In case of lines with "loop_" the category is in the
                        # next line
                        category_in_line = get_category_name(self._lines[i+1])
                    is_loop = is_loop_in_line
                    current_category = category_in_line
                    start = i
        # Add the entry for the final category
        # Since at the end of the file the end of the category
        # is not determined by the start of a new one,
        # this needs to be handled separately
        stop = len(self._lines)
        self._add_category(data_block, current_category,
                           start, stop, is_loop)
    
    def get_block_names(self):
        return list(self._data_blocks.keys())
    
    def get_category(self, block, category):
        category_info = self._data_blocks[block][category]
        start = category_info[0]
        stop = category_info[1]
        is_loop = category_info[2]
        
        category_dict = {}
        line_i = start
        
        if is_loop:
            line_i += 1
            in_array_data = False
            while line_i < stop:
                line = self._lines[line_i]
                if not _is_empty(line):
                    if line[0] != "_":
                        in_array_data = False
                    if not in_array_data:
                        pass
                    else:
                        pass
                line_i += 1
        
        else:
            while line_i < stop:
                line = self._lines[line_i]
                if not _is_empty(line):
                    parts = shlex.split(line)
                line_i += 1
            
    
    def set_category(self, block, category, data):
        pass
        
    def write(self, file_name):
        pass
    
    def copy(self):
        PDBxFile = PDBxFile()
        PDBxFile._lines = copy.deepcopy(self._lines)
        PDBxFile._data_blocks = copy.deepcopy(self._data_blocks)
    
    def _add_data_block(self, block_name):
        self._data_blocks[block_name] = {}
        return self._data_blocks[block_name]
        
    def _add_category(self, block, category_name, start, stop, is_loop):
        # Before the first category starts,
        # the current_category is None
        # This is checked before adding an entry
        if category_name is not None:
            block[category_name] = (start, stop, is_loop)


def _is_empty(line):
    return len(line) == 0 or line[0] == "#"


def _data_block_name(line):
    if line.startswith("data_"):
        return line[5:]
    else:
        return None

def _is_loop_start(line):
    return line.startswith("loop_")


def _is_data_header(line):
    pass


def get_category_name(line):
    if line[0] != "_":
        return None
    else:
        return line[1:line.find(".")]
    