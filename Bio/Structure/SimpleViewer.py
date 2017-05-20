# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from . import AtomArray

class SimpleViewer(object):
    
    def __init__(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        self._fig = plt.figure()
        self._ax = self._fig.gca(projection='3d')
        self._ax.axis("off")
        self._ax.set_aspect("equal")
        self._ax.set_xlim(-10,10)
        self._ax.set_ylim(-10,10)
        self._ax.set_zlim(-10,10)
        self._fig.tight_layout()
        
    def visualise(self, atom_array: AtomArray, color=None):
        self._ax.plot(atom_array.pos[:,0],
                      atom_array.pos[:,1],
                      atom_array.pos[:,2],
                      color=color)
        
    def show(self):
        import matplotlib.pyplot as plt
        plt.show()