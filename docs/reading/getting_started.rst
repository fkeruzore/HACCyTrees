Getting Started 
===============

.. currentmodule:: haccytrees.mergertrees


.. _sec-reading-a-forest:

Reading a Merger Forest
-----------------------

Merger forests are stored in the HDF5 format, and although one could read the HDF5
file directly, the :meth:`read_forest` function in haccytrees provides more
convenience, e.g. by creating additional indices that make walking the tree much
easier and by allowing to split the forest files into self-contained chunks.

For example, if we want to split a Last Journey merger forest into 10 chunks and
read the first chunk:

.. code-block:: python

   import numpy as np
   import haccytrees.mergertrees

   forest, progenitor_array = haccytrees.mergertrees.read_forest(
       "/data/a/cpac/mbuehlmann/LastJourney/m000p.forest.000.hdf5", 
       simulation="LastJourney", 
       nchunks=10, chunknum=0)

The returned `forest` is a dictionary containing one-dimensional numpy arrays


Extracting Main-Branch Matrices
-------------------------------

Sometimes, only the main-branch (defined by following the most massive
progenitor at each timestep) is needed. The function :func:`get_mainbranch_indices`
is a convenient function to construct a matrix of shape `(n_targets x n_steps)`,
where each column corresponds to the main branch of a halo, and each row 
corresponds to an output step of the simulation. At times where a halo does not
exist, the index is set to -1.

This index matrix can then be used to get the history of any stored halo
parameter. As an example, we can easily extract the mass history of all halos in
the mass-bin [1e13, 1e14] at z=0:

.. code-block:: python

    z0_mask = forest['snap_num'] == 100
    mlim = [1e13, 2e13]
    target_mask = z0_mask & (forest['mass'] > mlim[0]) * (forest['mass'] < mlim[1])
    target_idx = forest['halo_index'][target_mask]
        
    # this will create a matrix of shape (ntargets, nsteps), where each column 
    # is the main progenitor branch of a target. It contains the indices to the 
    # forest data, and is -1 if the halo does not exist at that time
    mainbranch_index = haccytrees.mergertrees.get_mainbranch_indices(
        forest, simulation='LastJourney', target_index=target_idx
    )
        
    # Get the mass of the main branches
    active_mask = mainbranch_index != -1
    mainbranch_mass = np.zeros_like(mainbranch_index, dtype=np.float32)
    mainbranch_mass[active_mask] = forest['mass'][mainbranch_index[active_mask]]


Finding Major Mergers
---------------------

Another common task is finding mergers above a certain relative or absolute
threshold.

--------------------------------------------------------------------------------

References
----------

.. autofunction:: read_forest

.. autofunction:: get_mainbranch_indices