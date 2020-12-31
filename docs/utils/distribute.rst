MPI Distribution Algorithms
===========================

.. currentmodule:: haccytrees.utils.distribute

Processing large datasets on multiple MPI ranks requires to distribute the data
among the processes. The ``haccytrees`` package contains the following functions 
that abstract this task in different use-cases:

.. autosummary::
   :nosignatures:

   distribute
   overload
   exchange

Examples
--------


References
----------

distribute
^^^^^^^^^^
.. autofunction:: distribute

overload
^^^^^^^^
.. autofunction:: overload

exchange
^^^^^^^^
.. autofunction:: exchange