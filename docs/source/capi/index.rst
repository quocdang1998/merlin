C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}

Basic functionality
-------------------

Printing log messages
^^^^^^^^^^^^^^^^^^^^^

Macro functions for printing log messages and throwing an exception:

.. doxysummary::
   :toctree: generated

   MESSAGE
   WARNING
   FAILURE
   CUDAOUT
   CUDAERR
   CUHDERR

Exception classes reserved for errors related to CUDA:

.. doxysummary::
   :toctree: generated

   cuda_compile_error
   cuda_runtime_error

1D vector
^^^^^^^^^

One dimensional sequence of data:

.. doxysummary::
   :toctree: generated

   merlin::Vector
   merlin::intvec

File mutex
^^^^^^^^^^

Lock for preventing data-race when reading or writing a file:

.. doxysummary::
   :toctree: generated

   merlin::FileLock

Utils
^^^^^

Get system information:

.. doxysummary::
   :toctree: generated

   merlin::get_current_process_id
   merlin::get_time

Flatten loop on multi-dimensional array:

.. doxysummary::
   :toctree: generated

   merlin::inner_prod
   merlin::ndim_to_contiguous_idx
   merlin::contiguous_to_ndim_idx


GPU with CUDA
-------------

CUDA Runtime API Wrapper
^^^^^^^^^^^^^^^^^^^^^^^^

C++ wrapper classes for CUDA runtime API and CUDA driver API:

.. doxysummary::
   :toctree: generated

   merlin::cuda::Device
   merlin::cuda::Context
   merlin::cuda::Event
   merlin::cuda::Stream
   merlin::cuda::record_event

GPU query
^^^^^^^^^

Print and test the compatibility of GPU and CUDA driver:

.. doxysummary::
   :toctree: generated

   merlin::cuda::print_all_gpu_specification
   merlin::cuda::test_all_gpu

Context management
^^^^^^^^^^^^^^^^^^

Print and test the compatibility of GPU and CUDA driver:

.. doxysummary::
   :toctree: generated

   merlin::cuda::default_context
   merlin::cuda::create_primary_context

Array API
---------

Multi-dimensional array
^^^^^^^^^^^^^^^^^^^^^^^

Classes represent multi-dimensional array on CPU, out-of-core array and GPU
array:

.. doxysummary::
   :toctree: generated

   merlin::array::NdData
   merlin::array::Array
   merlin::array::Parcel
   merlin::array::Stock

Array manipulation
^^^^^^^^^^^^^^^^^^

Utils for array manipulation:

.. doxysummary::
   :toctree: generated

   merlin::array::Slice
   merlin::array::array_copy

Grid API
--------

.. doxysummary::
   :toctree: generated

   merlin::interpolant::Grid
   merlin::interpolant::RegularGrid
   merlin::interpolant::CartesianGrid

