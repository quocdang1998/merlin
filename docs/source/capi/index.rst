C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}


Environment
-----------

.. doxysummary::
   :toctree: generated

   merlin::Environment
   merlin::default_environment


GPU with CUDA
-------------

Device Management
^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::cuda::Device
   merlin::cuda::Context
   merlin::cuda::print_all_gpu_specification
   merlin::cuda::test_all_gpu

Concurrency
^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::cuda::Event
   merlin::cuda::Stream
   merlin::cuda::GraphNode
   merlin::cuda::Graph
   merlin::cuda::begin_capture_stream
   merlin::cuda::end_capture_stream

Enum types
^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::cuda::DeviceLimit
   merlin::cuda::ContextSchedule
   merlin::cuda::EventCategory
   merlin::cuda::EventWaitFlag
   merlin::cuda::MemcpyKind
   merlin::cuda::NodeType
   merlin::cuda::StreamSetting

Array API
---------

1D vector
^^^^^^^^^

One dimensional sequence of data:

.. doxysummary::
   :toctree: generated

   merlin::Vector
   merlin::intvec

Multi-dimensional array
^^^^^^^^^^^^^^^^^^^^^^^

Classes represent multi-dimensional array on CPU, out-of-core array and GPU array:

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


Interpolant API
---------------

Grid
^^^^

.. doxysummary::
   :toctree: generated

   merlin::interpolant::Grid
   merlin::interpolant::RegularGrid
   merlin::interpolant::CartesianGrid
   merlin::interpolant::SparseGrid

Polynomial interpolant
^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::interpolant::PolynomialInterpolant
   merlin::interpolant::Method


Statistics API
--------------

.. doxysummary::
   :toctree: generated

   merlin::statistics::powered_mean
   merlin::statistics::moment_cpu


Low level API
-------------

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

File mutex
^^^^^^^^^^

Lock for preventing data-race when reading or writing a file:

.. doxysummary::
   :toctree: generated

   merlin::FileLock
