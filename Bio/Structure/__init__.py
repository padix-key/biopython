"""
A subpckage for handling protein structures. 

This subpackage enables efficient and easy handling of protein structure data
by representation of atom properties in `numpy` arrays.
This approach has multiple advantages:
    
    - Convenient selection of atoms in a structure
      by using `numpy` style indexing
    - Fast calculations on structures using C-accelerated `ndarray` operations
    - Simple implementation of custom calculations
"""

from .Atoms import *
from .Util import *
from .Error import *

from .Geometry import *
from .Compare import *

from .Transform import *
from .Superimpose import *

from .Vis import *