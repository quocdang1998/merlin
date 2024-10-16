C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}

In this documentation, the function and class method decorator ``__host__ __device__`` annotates a callable on both CPU
and GPU, whereas the decorator ``__device__`` indicates that the callable is available only on GPU. Functions and class
methods without any annotations can be executed exclusively by CPU.

In case of non-CUDA compilation, ``__host__ __device__`` functions will becomes regular CPU function, while
``__device__`` functions are discarded in preprocessing step.

Preliminary
-----------

Environment
^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::Environment

Vector types
^^^^^^^^^^^^

Template classes
''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::vector::View
   merlin::vector::StaticVector
   merlin::vector::DynamicVector

Type aliases
''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::UIntView
   merlin::IntView
   merlin::DoubleView

.. doxysummary::
   :toctree: generated

   merlin::UIntVec
   merlin::IntVec
   merlin::DoubleVec

.. doxysummary::
   :toctree: generated

   merlin::max_dim
   merlin::Index
   merlin::Point
   merlin::DPtrArray

.. doxysummary::
   :toctree: generated

   merlin::DViewArray
   merlin::DVecArray

Grid API
^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::grid::CartesianGrid
   merlin::grid::RegularGrid

Asynchronous launch
^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::ProcessorType
   merlin::Synchronizer


GPU with CUDA
-------------

Device management
^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::cuda::Device
   merlin::cuda::print_gpus_spec
   merlin::cuda::test_all_gpu

Asynchronous operations
^^^^^^^^^^^^^^^^^^^^^^^

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

Memory management helper
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::cuda::Dispatcher
   merlin::cuda::copy_objects


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

Polynomial
^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::regpl::Polynomial

Constructor and Evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::regpl::Vandermonde
   merlin::regpl::Regressor


Linear algebra API
------------------

Vector-vector operations
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::saxpy
   merlin::linalg::dot
   merlin::linalg::norm
   merlin::linalg::normalize
   merlin::linalg::householder

Matrix
^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::Matrix

Triangular solver
^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::triu_one_solve
   merlin::linalg::triu_solve

QR decomposition
^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::linalg::QRPDecomp

Canonical decomposition API
---------------------------

CP decomposition model
^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Model
   merlin::candy::Gradient

Model initialization
^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Randomizer
   merlin::candy::rand::Gaussian
   merlin::candy::rand::Uniform

Metric error
^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::TrainMetric
   merlin::candy::rmse_cpu
   merlin::candy::rmae_cpu
   merlin::candy::rmse_gpu
   merlin::candy::rmae_gpu

Model training algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::candy::Optimizer

Gradient descent
''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::GradDescent
   merlin::candy::optmz::create_grad_descent

Adaptative gradient
'''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::AdaGrad
   merlin::candy::optmz::create_adagrad

Adaptive estimates of lower-order moments
'''''''''''''''''''''''''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::Adam
   merlin::candy::optmz::create_adam

Adaptive delta
''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::AdaDelta
   merlin::candy::optmz::create_adadelta

Root mean square propagation
''''''''''''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::optmz::RmsProp
   merlin::candy::optmz::create_rmsprop

Launch calculation
^^^^^^^^^^^^^^^^^^

Generalized launch
''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::Trainer
   merlin::candy::TrialPolicy

CPU/GPU specified API
'''''''''''''''''''''

.. doxysummary::
   :toctree: generated

   merlin::candy::train::TrainerBase
   merlin::candy::train::CpuTrainer
   merlin::candy::train::GpuTrainer

Low level API
-------------

Memory allocation
^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::mem_alloc_host
   merlin::mem_free_host
   merlin::mem_register_host
   merlin::mem_unregister_host

.. doxysummary::
   :toctree: generated

   merlin::mem_alloc_device
   merlin::mem_free_device

.. doxysummary::
   :toctree: generated

   merlin::memcpy_cpu_to_gpu
   merlin::memcpy_gpu_to_cpu
   merlin::memcpy_gpu
   merlin::memcpy_peer_gpu

Log printing
^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::Message
   merlin::Warning
   merlin::Fatal
   merlin::CudaOut
   merlin::DeviceError
   merlin::DebugLog

.. doxysummary::
   :toctree: generated

   merlin::cuda_compile_error
   merlin::cuda_runtime_error

IO operations
^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::io::FileDeleter
   merlin::io::FilePointer
   merlin::io::open_file

.. doxysummary::
   :toctree: generated

   merlin::io::FileLock

.. doxysummary::
   :toctree: generated

   merlin::io::IoBuffer
   merlin::io::ReadEngine
   merlin::io::WriteEngine

Get system info
^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::get_current_process_id
   merlin::get_time

Vectorization
^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::AvxFlag
   merlin::use_avx
   merlin::AvxDouble

CUDA thread index
^^^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::flatten_thread_index
   merlin::size_of_block
   merlin::flatten_block_index
   merlin::flatten_kernel_index

Permutation
^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::Permutation
