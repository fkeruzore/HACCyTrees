Creating Merger Forests
=======================

Outline
-------

.. automodule:: treenodes2forest

Configuration
-------------

.. literalinclude:: ../../executables/treenodes2forest.example.cfg


Example: Last Journey 
---------------------

Creating the :ref:`sim_LastJourney` Merger Forest took 3.2 hours on cooley, using
32 nodes with 12 MPI ranks each (total of 384 ranks). The majority of the time
was spent in reading the treenode files (1.3 hours) and doing temporary IO (1h)
in order to keep the memory consumption in each rank low. Indexing and reordering
the data comparably took an insignificant amount of time.


.. image:: LJ_timing_32nodes.svg 
   :width: 100%

References
----------

.. autofunction:: haccytrees.mergertrees.assemble.catalog2tree