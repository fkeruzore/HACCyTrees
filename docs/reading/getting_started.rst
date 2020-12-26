Getting Started 
===============

.. currentmodule:: haccytrees.mergertrees




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
corresponds to an output step of the simulation.



--------------------------------------------------------------------------------

References
----------

.. autofunction:: read_forest

.. autofunction:: get_mainbranch_indices