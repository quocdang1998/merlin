# Copyright 2022 quocdang1998

cdef extern from "merlin/cuda/context.hpp":

    cpdef enum class ContextFlags "merlin::cuda::Context::Flags":
        """
        CUDA Context setting flags.

        *Values*

         - ``AutoSchedule``: Automatic schedule based on the number of context and number of logical process.
         - ``SpinSchedule``: Actively spins when waiting for results from the GPU.
         - ``YieldSchedule``: Yield the CPU process when waiting for results from the GPU.
         - ``BlockSyncSchedule``: Block CPU process until synchronization.
        """
        AutoSchedule,
        SpinSchedule,
        YieldSchedule,
        BlockSyncSchedule

    cdef cppclass CppContext "merlin::cuda::Context":
        CppContext()
        CppContext(const CppDevice & gpu, ContextFlags flag) except +
        CppContext(const CppContext & src) except +
        uintptr_t get_context_ptr()
        CppDevice get_gpu() except +
        bint is_attached() except +
        void increase_reference_count() except +
        void decrease_reference_count() except +
        void push_current() except +
        CppContext & pop_current() except +
        bint is_current() except +
        void set_current() except +
        string repr() except +

        @staticmethod
        CppContext get_current() except +
