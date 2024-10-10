Developer Guide
===============

Repository hierarchy
--------------------

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


Cautions
--------

Some cautions when developing the package:

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   ============================= ========================================================
   :doc:`mutex`                  List of functions altering the mutex lock guard.
   ============================= ========================================================

.. toctree::
   :maxdepth: -1
   :hidden:

   mutex


Data serialization
------------------

Data in Merlin can be serialized into a binary data file for transferring data across multiple executables. There are
three classes that can be serialized/deserialized in Merlin: multi-dimensional array, regression polynomial and
CANDECOMP-PARAFAC model.

By convention, integers and floating points utilized by Merlin are in **8 bytes**. For integer values, the fixed-width
type ``std::int64_t`` and ``std::uint64_t`` is preferred over ``int``, ``unsigned int`` or ``std::size_t``. Similarly,
for floating points, the usage of ``double`` is preferred over ``float`` (to be converted to ``std::float64_t`` when
CUDA officially supports the C++23 standard). Conventional C-types are only utilized at the interface with other
libraries, such as CUDA.

Data are serialized in **little endian**. Conversions between little endian and native endian are automatically handled
by the read engine :cpp:class:`merlin::io::ReadEngine` and the write engine :cpp:class:`merlin::io::WriteEngine`.

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   ============================= ========================================================
   :doc:`array`                  Serialization algorithm of multi-dimensional array
   :doc:`polynomial`             Serialization algorithm of regression polynomial
   :doc:`cpmodel`                Serialization algorithm of CP model
   ============================= ========================================================

.. toctree::
   :maxdepth: -1
   :hidden:

   array
   polynomial
   cpmodel

Developer of other projects can read and manipulate data in the serialized file either through the Merlin's API, or
directly using C/C++ IO library such as ``std::fread`` / ``std::fwrite`` or ``std::istream::read`` /
``std::ostream::write``.
