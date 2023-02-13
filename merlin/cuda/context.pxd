# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t, uintptr_t
from libcpp.string cimport string

from merlin.cuda.device cimport CppDevice

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
        uint64_t get_reference_count()
        bint is_primary()
        void push_current() except +
        CppContext & pop_current() except +
        bint is_current() except +
        string str() except +

        @staticmethod
        CppContext get_current() except +
        @staticmethod
        CppDevice get_gpu_of_current_context() except +
        @staticmethod
        ContextFlags get_flag_of_current_context() except +
        @staticmethod
        void synchronize() except +

    bint operator==(const CppContext & ctx_1, const CppContext & ctx_2)
    bint operator!=(const CppContext & ctx_1, const CppContext & ctx_2)
    CppContext cpp_create_primary_context "merlin::cuda::create_primary_context" (const CppDevice & gpu, ContextFlags flag) except +