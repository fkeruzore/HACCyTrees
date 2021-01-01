HACCYTREES
==========

[add some summary]

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

   # Altenratively, if you want to be able to edit the code without having
   # to reinstall the library
   python setup.py develop


Requirements
------------

These python packages are required to be installed:

- numpy
- numba
- h5py

Some parts of the library require the following packages:

- drawSvg
- matplotlib 
- mpi4py
- pygio


About
-----