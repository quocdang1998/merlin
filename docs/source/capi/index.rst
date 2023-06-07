C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}


Preliminary
-----------

Environment
^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::Environment
   merlin::default_environment

One dimensional sequence of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::Vector
   merlin::intvec
   merlin::floatvec

Shuffle elements
^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::Shuffle

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

Multi-dimensional array
^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::array::NdData
   merlin::array::Array
   merlin::array::Parcel
   merlin::array::Stock

Array manipulation
^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::array::Slice
   merlin::array::array_copy
   merlin::array::shuffle_array
   merlin::array::shuffled_read


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

Statistical moments
^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::statistics::powered_mean
   merlin::statistics::moment_cpu

Linear algebra
^^^^^^^^^^^^^^

Vector inner product
^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::inner_product
   merlin::linalg::norm
   merlin::linalg::normalize

Canonical decomposition API
---------------------------

CP decomposition model
^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Model
   merlin::candy::RandomInitializer
   # merlin::candy::calc_loss_function_cpu
   # merlin::candy::calc_loss_function_gpu
   # merlin::candy::calc_gradient_vector_cpu
   # merlin::candy::calc_gradient_vector_gpu

Model training algorithm
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Optimizer
   merlin::candy::optmz::GradDescent

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

.. doxysummary::
   :toctree: generated

   merlin::FileLock

Get system information
^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::get_current_process_id
   merlin::get_time

CUDA kernel thread index
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::flatten_thread_index
   merlin::size_of_block
   merlin::flatten_block_index
   merlin::flatten_kernel_index
