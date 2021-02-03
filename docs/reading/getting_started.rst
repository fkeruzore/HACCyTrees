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
threshold. To get the merger ratio from the two most massive progenitors of a
list of target halos, the function :func:`get_nth_progenitor_indices` can be
used as follows:

.. code-block:: python

   # get indices to main progenitors
   main_progenitor_index = haccytrees.mergertrees.get_nth_progenitor_indices(
       forest, progenitor_array, target_index=mainbranch_index[active_mask], n=1
   )

   # get indices to secondary progenitors (main mergers)
   main_merger_index = haccytrees.mergertrees.get_nth_progenitor_indices(
       forest, progenitor_array, target_index=mainbranch_index[active_mask], n=2
   )

   # the index will be negative if there's no merger, mask those out
   merger_mask = main_merger_index >= 0

   # allocate a merger_ratio array, 0 by default
   merger_ratio = np.zeros_like(main_progenitor_index, dtype=np.float32)

   # fill the elements for which a merger occurred with the mass ratio
   merger_ratio[merger_mask] = forest['tree_node_mass'][main_merger_index[merger_mask]] / 
                               forest['tree_node_mass'][main_progenitor_index[merger_mask]]
   
Major mergers can then be identified by finding the entries in ``merger_ratio``
above the major merger threshold. 

If an absolute major merger criteria is required, we only have to extract the
mass of the main merger (secondary progenitor), i.e.

.. code-block:: python

   # get indices to secondary progenitors (main mergers)
   main_merger_index = haccytrees.mergertrees.get_nth_progenitor_indices(
       forest, progenitor_array, target_index=mainbranch_index[active_mask], n=2
   )

   # the index will be negative if there's no merger, mask those out
   merger_mask = main_merger_index >= 0

   # allocate a merger_ratio array, 0 by default
   merger_mass = np.zeros_like(main_progenitor_index, dtype=np.float32)

   # fill the elements for which a merger occurred with the mass ratio
   merger_mass[merger_mask] = forest['tree_node_mass'][main_merger_index[merger_mask]] 


--------------------------------------------------------------------------------

References
----------

.. autofunction:: read_forest

.. autofunction:: get_mainbranch_indices

.. autofunction:: get_nth_progenitor_indices