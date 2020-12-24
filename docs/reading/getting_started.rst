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

.. code-block:: python

    # creating a target mask, e.g. halos with masses > 1e13 Msun/h
    target_mask = (forest['snap_num'] == 100) & (forest['mass'] > 1e13)
    target_idx = forest['halo_index'][target_mask]
    
    # this will create a matrix of shape (ntargets, nsteps), where each column 
    # is the main progenitor branch of a target. It contains the indices to the 
    # forest data, and is -1 if the halo does not exist at that time
    mainbranch_index = haccytrees.mergertrees.get_mainbranch_indices(
        forest, simulation='LastJourney', target_index=target_idx
    )
    
    # Example: create the mass history for each target
    active_mask = mainbranch_index != -1
    mainbranch_mass = np.zeros_like(mainbranch_index, dtype=np.float32)
    mainbranch_mass[active_mask] = forest['mass'][mainbranch_index[active_mask]]
    
    # this is just to get the scale factors associated with each step (matrix row)
    simulation = haccytrees.simulation_lut['LastJourney']
    scale_factors = simulation.step2a(np.array(simulation.cosmotools_steps))
    
    # plotting the average mass evolution for the halos in the mass bin
    fig, ax = plt.subplots()
    ax.plot(scale_factors, np.mean(mainbranch_mass, axis=0))
    ax.set(
        yscale='log', 
        xlabel='scale factor $a$', 
        ylabel=r'$\langle M_\mathrm{FoF} \rangle$ [$h^{-1}M_\odot$'
    )

--------------------------------------------------------------------------------

References
----------

.. autofunction:: read_forest

.. autofunction:: get_mainbranch_indices

.. autofunction:: split_fragment_tag