C++ API
=======

.. raw:: latex

   \setcounter{codelanguage}{1}


Environment
-----------

Execution environment:

.. doxysummary::
   :toctree: generated

   merlin::Environment
   merlin::default_environment


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

Lagrange method
^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::interpolant::calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid &grid, const array::Array &value, array::Array &coeff) "calc_lagrange_coeffs_cpu_on_cartgrid"
   ~merlin::interpolant::calc_lagrange_coeffs_gpu "calc_lagrange_coeffs_gpu"
   merlin::interpolant::calc_lagrange_coeffs_cpu(const interpolant::SparseGrid &grid, const array::Array &value, array::Array &coeff) "calc_lagrange_coeffs_cpu_on_sparsegrid"
   merlin::interpolant::eval_lagrange_cpu(const interpolant::CartesianGrid &, const array::Array &, const Vector<double> &) "eval_lagrange_cpu_on_cartgrid"
   ~merlin::interpolant::eval_lagrange_gpu "eval_lagrange_gpu"
   merlin::interpolant::eval_lagrange_cpu(const interpolant::SparseGrid &, const array::Array &, const Vector<double> &) "eval_lagrange_cpu_on_sparsegrid"

Newton method
^^^^^^^^^^^^^^^

.. doxysummary::
   :toctree: generated

   merlin::interpolant::calc_newton_coeffs_cpu(const interpolant::CartesianGrid &grid, const array::Array &value, array::Array &coeff) "calc_newton_coeffs_cpu_on_cartgrid"
   merlin::interpolant::calc_newton_coeffs_gpu "calc_newton_coeffs_gpu"
   merlin::interpolant::calc_newton_coeffs_cpu(const interpolant::SparseGrid &grid, const array::Array &value, array::Array &coeff) "calc_newton_coeffs_cpu_on_sparsegrid"
   merlin::interpolant::eval_newton_cpu(const interpolant::CartesianGrid &, const array::Array &, const Vector<double> &) "eval_newton_cpu_on_cartgrid"
   merlin::interpolant::eval_newton_gpu "eval_newton_gpu"
   merlin::interpolant::eval_newton_cpu(const interpolant::SparseGrid &, const array::Array &, const Vector<double> &) "eval_newton_cpu_on_sparsegrid"


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
