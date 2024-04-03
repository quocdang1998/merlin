C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}

In this documentation, the function and class method decorator ``__host__ __device__`` annotates a callable on both CPU
and GPU, whereas the decorator ``__device__`` indicates the callable is available only on GPU. Functions and class
methods without any annotations can be executed exclusively by CPU.

In case of non-CUDA compilation, ``__host__ __device__`` functions will becomes regular CPU function, while
``__device__`` functions are discarded in preprocessing step.

Preliminary
-----------

.. doxysummary::
   :toctree: generated

   merlin::Environment

Type aliasing
-------------

.. doxysummary::
   :toctree: generated

   merlin::Vector
   merlin::UIntVec
   merlin::DoubleVec

.. doxysummary::
   :toctree: generated

   merlin::max_dim
   merlin::Index
   merlin::Point
   merlin::DPtrArray

GPU with CUDA
-------------

Device Management
^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::cuda::Device
   merlin::cuda::print_gpus_spec
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

   merlin::array::copy
   merlin::array::fill
   merlin::array::stat
   merlin::array::print

Grid API
--------

.. doxysummary::
   :toctree: generated

   merlin::grid::CartesianGrid
   merlin::grid::RegularGrid


Interpolator API
----------------

Polynomial interpolation
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::splint::Interpolator
   merlin::splint::Method

Low-level API
^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::splint::construct_coeff_cpu
   merlin::splint::construct_coeff_gpu
   merlin::splint::eval_intpl_cpu
   merlin::splint::eval_intpl_gpu


Regression API
--------------

.. doxysummary::
   :toctree: generated

   merlin::regpl::Polynomial
   merlin::regpl::Regressor

Linear algebra API
------------------

Vector inner product
^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::dot
   merlin::linalg::norm
   merlin::linalg::householder

Solving linear system by QR decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::Matrix

.. doxysummary::
   :toctree: generated

   merlin::linalg::qr_decomposition_cpu
   merlin::linalg::qr_solve_cpu

.. doxysummary::
   :toctree: generated

   merlin::linalg::qr_decomposition_gpu
   merlin::linalg::qr_solve_gpu

Canonical decomposition API
---------------------------

CP decomposition model
^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Model
   merlin::candy::Gradient

Metric error
^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::TrainMetric
   merlin::candy::rmse_cpu
   merlin::candy::rmae_cpu
   merlin::candy::rmse_gpu
   merlin::candy::rmae_gpu

Model training algorithm
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Optimizer

Gradient descent
''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::GradDescent
   merlin::candy::create_grad_descent

Adaptative gradient
'''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::AdaGrad
   merlin::candy::create_adagrad

Adaptive estimates of lower-order moments
'''''''''''''''''''''''''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::Adam
   merlin::candy::create_adam

Launch calculation
^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Trainer

Low level API
-------------

.. doxysummary::
   :toctree: generated

   MESSAGE
   WARNING
   FAILURE
   CUDAOUT
   CUDAERR
   CUHDERR

.. doxysummary::
   :toctree: generated

   merlin::FileLock

.. doxysummary::
   :toctree: generated

   merlin::get_current_process_id
   merlin::get_time

.. doxysummary::
   :toctree: generated

   merlin::flatten_thread_index
   merlin::size_of_block
   merlin::flatten_block_index
   merlin::flatten_kernel_index

.. doxysummary::
   :toctree: generated

   merlin::Permutation
