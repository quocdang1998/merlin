C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}

Array API
---------

Basic utils for Multi-dimensional array manipulation:

.. doxysummary::
   :toctree: generated

   merlin::Vector
   merlin::intvec
   merlin::Iterator
   merlin::NdData
   merlin::Slice

Classes represent multi-dimensional array on CPU, out-of-core array and GPU
array:

.. doxysummary::
   :toctree: generated

   merlin::Array
   merlin::Parcel
   merlin::Stock

Grid API
--------

.. doxysummary::
   :toctree: generated

   merlin::Grid
   merlin::RegularGrid
   merlin::CartesianGrid


Log API
-------

Macro functions for printing log messages and throwing an exception:

.. doxysummary::
   :toctree: generated

   MESSAGE
   WARNING
   FAILURE
   CUDAOUT

Exception classes reserved for errors related to CUDA:

.. doxysummary::
   :toctree: generated

   cuda_compile_error
   cuda_runtime_error

Utils
-----

.. doxysummary::
   :toctree: generated

   merlin::inner_prod
   merlin::ndim_to_contiguous_idx
   merlin::contiguous_to_ndim_idx
   merlin::array_copy

