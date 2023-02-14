Developer Guide
===============

C++ sources
-----------


Source files of C++ API are placed in folder ``src/merlin``. Each file extension
corresponds to a specific role as described in the image below.

.. literalinclude:: ../_img/source_tree.txt
   :language: sh

For more specific information, consult the sections below:

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   ============================= ========================================================
   :doc:`compilation`            The way Cmake configure the compilation of the package.
   :doc:`macro`                  The role of each macro used in the source.
   :doc:`style`                  The code style rules of the source code.
   ============================= ========================================================

.. toctree::
   :maxdepth: -1
   :hidden:

   compilation
   macro
   style
