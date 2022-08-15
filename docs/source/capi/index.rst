C++ API
=======

Array API
---------

This API allow manipulation with multi-dimensional array CPU, out-of-core and
GPU array:

.. doxysummary::
   :toctree: generated

   merlin::Array
   merlin::Tensor
   merlin::Parcel

Some classes facilitates array abstraction:

.. doxysummary::
   :toctree: generated

   merlin::Slice


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
