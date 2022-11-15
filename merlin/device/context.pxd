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
        CppContext(const CppDevice & gpu, ContextFlags flag)
        CppContext(const CppContext & src)
        CppDevice get_gpu()
        bint is_attached()
        void push_current()
        CppContext & pop_current()
        bint is_current()
        void set_current()

        @staticmethod
        CppContext & get_primary_context(const CppDevice & gpu)
        @staticmethod
        pair[bint, ContextFlags] get_primary_ctx_state(const CppDevice & gpu)
