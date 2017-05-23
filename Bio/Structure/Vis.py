# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

def simple_view(fig, atom_arrays):
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.gca(projection='3d')
    ax.axis("off")
    ax.set_aspect("equal")
    # Sets equal limits to all dimension, since set_aspect("equal") does not work properly
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    for atom_array in atom_arrays:
        ax.plot(atom_array.pos[:,0],
                 atom_array.pos[:,1],
                 atom_array.pos[:,2],)
    return ax