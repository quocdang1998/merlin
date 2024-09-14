Mutex
=====

Mutex is quintessential for guarding the critical zone inside a parallel region for the purpose of atomic operation.
However, mutex must lock must be acquired carefully. Otherwise, it will lead to deadlock, a situation in which the
program is stuck inside an infinite loop.

Warning about the mutex in the documentation of functions locking or unlocking the mutex is mandatory.

This section lists all functions and class methods that interact with the mutex of
:cpp:member:`Environment::mutex <merlin::Environment::mutex>`. Developers must pay attention to avoid these functions
inside a critical region. When compiling Merlin in debug mode (i.e. ``CMAKE_BUILD_TYPE=Debug``), a notification and
stacktrace is printed out to the standard output whenever the mutex is locked or unlocked. They provide information of
the source location at which the mutex is altered, thus facilitating deadlock traceback.

CUDA library
------------

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   =========================================================================== =========================================
   :cpp:func:`cuda::Device::push_context <merlin::cuda::Device::push_context>` Lock the mutex without releasing.
   :cpp:func:`cuda::Device::pop_context <merlin::cuda::Device::pop_context>`   Only releasing the mutex without locking.
   :cpp:func:`cuda::Graph::execute <merlin::cuda::Graph::execute>`             CUDA context guard.
   =========================================================================== =========================================

ARRAY library
-------------

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   ========================================================================================= ===================
   :cpp:func:`array::Array::clone_data_from_gpu <merlin::array::Array::clone_data_from_gpu>` CUDA context guard.
   :cpp:func:`array::Parcel::get <merlin::array::Parcel::get>`                               CUDA context guard.
   :cpp:func:`array::Parcel::set <merlin::array::Parcel::set>`                               CUDA context guard.
   :cpp:func:`array::Parcel::free_current_data <merlin::array::Parcel::free_current_data>`   CUDA context guard.
   ========================================================================================= ===================

SPLINT library
--------------

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   ======================================================================================================= ===================
   :cpp:func:`splint::Interpolator::Interpolator <merlin::splint::Interpolator::Interpolator>`             CUDA context guard.
   :cpp:func:`splint::Interpolator::build_coefficients <merlin::splint::Interpolator::build_coefficients>` CUDA context guard.
   :cpp:func:`splint::Interpolator::evaluate <merlin::splint::Interpolator::evaluate>`                     CUDA context guard.
   :cpp:func:`splint::Interpolator::~Interpolator <merlin::splint::Interpolator::~Interpolator>`           CUDA context guard.
   ======================================================================================================= ===================

CANDY library
-------------

.. tabularcolumns:: \X{1}{2}\X{1}{2}

.. table::
   :class: longtable
   :widths: 30 70

   ============================================================================ ==========================================
   :cpp:class:`candy::trainer::GpuTrainer <merlin::candy::trainer::GpuTrainer>` CUDA context guard.
   :cpp:class:`candy::Trainer::Trainer <merlin::candy::Trainer>`                CUDA context guard (in GPU configuration).
   ============================================================================ ==========================================
