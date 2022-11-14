# Copyright 2022 quocdang1998

cdef extern from "merlin/device/gpu_query.hpp":

    cpdef enum class DeviceLimit "merlin::device::Device::Limit":
        """GPU limit flags.

        *Values*

         - ``StackSize``: Size of the stack of each CUDA thread.
         - ``PrintfSize``: Size of the ``std::printf`` function buffer.
         - ``HeapSize``: Size of the heap of each CUDA thread.``.
         - ``SyncDepth``: Maximum nesting depth of a grid at which a thread can safely call ``cudaDeviceSynchronize``.
         - ``LaunchPendingCount``: Maximum number of outstanding device runtime launches.
        """
        StackSize,
        PrintfSize,
        HeapSize,
        SyncDepth,
        LaunchPendingCount

    cdef cppclass CppDevice "merlin::device::Device":
        CppDevice() except +
        CppDevice(int id) except +
        CppDevice(const CppDevice & src) except +
        void print_specification() except +
        bint test_gpu() except +
        void set_as_current() except +
        string repr() except +
        int & id() except +

        @staticmethod
        CppDevice get_current_gpu() except +
        @staticmethod
        int get_num_gpu() except +
        @staticmethod
        uint64_t limit(DeviceLimit limit, uint64_t size) except +
        @staticmethod
        void reset_all() except+

    bint operator==(const CppDevice & left, const CppDevice & right) except +
    bint operator!=(const CppDevice & left, const CppDevice & right) except +

    void cpp_print_all_gpu_specification "merlin::device::print_all_gpu_specification" () except +
    bint cpp_test_all_gpu "merlin::device::test_all_gpu" () except +
