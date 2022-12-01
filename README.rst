HACCYTREES
==========

Welcome to haccytrees! This is a python library to create, read, and process
large merger-tree forests created from HACC simulations, such as Last Journey.

Documentation and usage examples are provided at
`<https://argonnecpac.github.io/haccytrees/>`_. The code is available at
`<https://github.com/ArgonneCPAC/haccytrees>`_.


.. image:: tree_example.svg
   :alt: merger tree illustration
   :width: 100%

.. _install-haccytrees:

Installation
------------

To install haccytrees, you will need to clone the repository and then use `pip`.

.. code-block:: bash

   git clone https://github.com/ArgonneCPAC/haccytrees.git
   cd haccytrees

   # Using pip to install the package
   pip install .


Requirements
------------

These python packages will be automatically installed if they are not yet in
your python library:

* `numpy <https://numpy.org/>`_: Python array library

* `numba <https://numba.pydata.org/>`_: used to speed up iterating across arrays
  and trees

* `h5py <https://www.h5py.org/>`_: a python HDF5 interface

For visualizations, these additional packages are required:

* `matplotlib <https://matplotlib.org/>`_: General purpose plotting library

* `drawSvg <https://github.com/cduck/drawSvg>`_: used to create SVG drawings of
  trees

These two packages are required to run the HACC to haccytrees conversion:

* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_: MPI for Python to
  distribute the work-load

* `pygio <https://git.cels.anl.gov/hacc/genericio>`_: The Python GenericIO
  interface that allows reading and writing GenericIO files from python with and
  without MPI. Use `this
  <https://git.cels.anl.gov/hacc/genericio/-/blob/master/new_python/README.md>`_
  guide to install ``pygio``.

