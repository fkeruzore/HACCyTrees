HACCYTREES
==========

Welcome to haccytrees! This is a python library to create, read, and process
large merger-tree forests created from HACC simulations, such as Last Journey.

Documentation and usage examples are provided at
`<https://www.hep.anl.gov/mbuehlmann/haccytrees>`_. The code is available at 
`<https://xgitlab.cels.anl.gov/mbuehlmann/haccytrees>`_.


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

   git clone https://xgitlab.cels.anl.gov/mbuehlmann/haccytrees.git
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

- numpy
- numba
- h5py

Some parts of the library require the following packages:

- drawSvg (for tree visualizations)
- matplotlib (for general plotting)
- mpi4py (for forest assembly)
- pygio (for forest assembly)
