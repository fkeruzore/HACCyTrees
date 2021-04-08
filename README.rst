HACCYTREES
==========

Welcome to haccytrees! This is a python library to create, read, and process
large merger-tree forests created from HACC simulations, such as Last Journey.

Documentation and usage examples are provided at
`<https://www.hep.anl.gov/mbuehlmann/haccytrees>`_. The code is available at 
`<https://git.cels.anl.gov/mbuehlmann/haccytrees>`_.


.. image:: tree_example.svg
   :alt: merger tree illustration
   :width: 100%

.. _install-haccytrees:

Installation
------------

Currently, haccytrees is hosted on the Argonne CELS gitlab. To install haccytrees,
you will need to clone the repository and then use `pip` or `python setup.py` to
install the library.

.. code-block:: bash

   git clone https://git.cels.anl.gov/mbuehlmann/haccytrees.git
   cd haccytrees

   # Using pip to install the package
   pip install .

   # Alternatively, using the setup.py installation script directly
   python setup.py install

   # Altenratively, if you want to be able to edit / update the code without 
   # having to reinstall the library
   python setup.py develop


Requirements
------------

These python packages are required to be installed:

* `numpy <https://numpy.org/>`_: Python array library

* `numba <https://numba.pydata.org/>`_: used to speed up iterating across arrays 
  and trees

* `h5py <https://www.h5py.org/>`_: a python HDF5 interface

For visualizations, these additional packages are required:

* `matplotlib <https://matplotlib.org/>`_: General purpose plotting library

* `drawSvg <https://github.com/cduck/drawSvg>`_: used to create SVG drawings of
  trees

To actually process treenode files and generate merger forests, two additional
packages are required:

* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_: MPI for Python to
  distribute the work-load

* `pygio <https://xgitlab.cels.anl.gov/hacc/genericio>`_: The Python GenericIO
  interface that allows reading and writing GenericIO files from python with
  and without MPI. Use `this <https://xgitlab.cels.anl.gov/hacc/genericio/-/blob/master/new_python/README.md>`_
  guide to install ``pygio``.
