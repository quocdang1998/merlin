# Copyright 2022 quocdang1998

cdef extern from "merlin/device/context.hpp":

    cpdef enum class ContextFlags "merlin::device::Context::Flags":
        """CUDA Context setting flags.

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

    cdef cppclass Cpp_Context "merlin::device::Context":
        Cpp_Context()
        Cpp_Context(const Cpp_Device & gpu, ContextFlags flag)
        Cpp_Context(const Cpp_Context & src)
        Cpp_Device get_gpu()
        bint is_attached()
