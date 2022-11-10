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

    cdef cppclass Cpp_Device "merlin::device::Device":
        Cpp_Device()
        Cpp_Device(int id)
        Cpp_Device(const Cpp_Device & src)
        void print_specification()
        bint test_gpu()
        void set_as_current()
        string repr()
        int & id()
    Cpp_Device cpp_Device_get_current_gpu "merlin::device::Device::get_current_gpu" ()
    int cpp_Device_get_num_gpu "merlin::device::Device::get_num_gpu" ()
    void cpp_Device_reset_all "merlin::device::Device::reset_all" ()
    uint64_t cpp_Device_limit "merlin::device::Device::limit" (DeviceLimit limit, uint64_t size)
    bint operator==(const Cpp_Device & left, const Cpp_Device & right)
    bint operator!=(const Cpp_Device & left, const Cpp_Device & right)

    void cpp_print_all_gpu_specification "merlin::device::print_all_gpu_specification" ()
    bint cpp_test_all_gpu "merlin::device::test_all_gpu" ()
