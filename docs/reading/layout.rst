Merger Forest Layout
====================

[Describe merger forest format / layout]


Available Columns
-----------------

.. table:: Available Columns
   :widths: 30 70
   :class: full-width
   
   +-----------------------+--------------------------------------------------------+
   |          Key          |                      Description                       |
   +=======================+========================================================+
   | tree_node_index       | the unique ID given to the halo in the treenode files  |
   +-----------------------+--------------------------------------------------------+
   | desc_node_index       | the tree_node_index of the descendant halo, -1 if no   |
   |                       | descendant                                             |
   +-----------------------+--------------------------------------------------------+
   | fof_halo_tag          | the FoF halo ID given by the in-situ halofinder.       |
   |                       | For a fragmented halo                                  |
   |                       | (:ref:`reading/fragments:Dealing with Fragments`),     |
   |                       | the tag is negative                                    |
   +-----------------------+--------------------------------------------------------+
   | snap_num              | the enumerated output, starting at 0 for the first     |
   |                       | snapshot                                               |
   +-----------------------+--------------------------------------------------------+
   | tree_node_mass        | the FoF mass of the halo, corrected for fragments      |
   +-----------------------+--------------------------------------------------------+
   | fof_halo_mass         | the FoF mass of the halo                               |
   +-----------------------+--------------------------------------------------------+
   | fof_halo_count        | the number of particles in the FoF                     |
   +-----------------------+--------------------------------------------------------+
   | sod_halo_mass         | the SOD mass (usually at 200c overdensity)             |
   +-----------------------+--------------------------------------------------------+
   | sod_halo_radius       | the SOD radius (usually at 200c overdensity)           |
   +-----------------------+--------------------------------------------------------+
   | sod_halo_cdelta       | the SOD concentration parameter                        |
   +-----------------------+--------------------------------------------------------+
   | sod_halo_cdelta_error | the error estimate of sod_halo_cdelta                  |
   +-----------------------+--------------------------------------------------------+
   | sod_halo_cdelta_accum | the SOD concentration parameter calculated using the   |
   |                       | accumulated mass algorithm                             |
   +-----------------------+--------------------------------------------------------+
   | sod_halo_cdelta_peak  | the SOD concentration parameter calculated using the   |
   |                       | dM/dr peak algorithm                                   |
   +-----------------------+--------------------------------------------------------+
   | xoff_fof              |                                                        |
   +-----------------------+--------------------------------------------------------+
   | xoff_sod              |                                                        |
   +-----------------------+--------------------------------------------------------+
   | xoff_com              |                                                        |
   +-----------------------+--------------------------------------------------------+


.. table:: Additional Columns
   :widths: 30 70
   :class: full-width
   
   +-------------------+----------------------------------------------------+
   |        Key        |                    Description                     |
   +===================+====================================================+
   | scale_factor      | scale factor                                       |
   +-------------------+----------------------------------------------------+
   | halo_idx          | the array position of the halo, created during the |
   |                   | reading of the forest file                         |
   +-------------------+----------------------------------------------------+
   | descendant_idx    | the array position of the descendant halo, created |
   |                   | during the reading of the forest file              |
   +-------------------+----------------------------------------------------+
   | progenitor_count  | the number of progenitor this halo has             |
   +-------------------+----------------------------------------------------+
   | progenitor_offset | the location in the ``progenitor_array`` where the |
   |                   | array indices to the progenitors are stored        |
   +-------------------+----------------------------------------------------+