# Copyright 2022 quocdang1998

cdef extern from "merlin/device/context.hpp":

    cpdef enum class ContextFlags "merlin::device::Context::Flags":
        AutoSchedule,
        SpinSchedule,
        YieldSchedule,
        BlockSyncSchedule

    cdef cppclass Cpp_device_Context "merlin::device::Context":
        Cpp_device_Context()
        Cpp_device_Context(const Cpp_device_Device & gpu, ContextFlags flag)
        Cpp_device_Device get_gpu()
