# Copyright 2022 quocdang1998

from libcpp.string cimport string

cdef extern from "merlin/device/gpu_query.hpp":
    int Cpp_device_get_current_gpu "merlin::device::get_current_gpu" ()

    cdef cppclass Cpp_device_Device "merlin::device::Device":
        Cpp_device_Device(int id)
        Cpp_device_Device(const Cpp_device_Device & src)
        Cpp_device_Device & operator=(const Cpp_device_Device & src)
        void print_specification()
        bint test_gpu()
        string repr()
    int cpp_device_Device_get_num_gpu "merlin::device::Device::get_num_gpu" ()
    void cpp_device_Device_reset_all "merlin::device::Device::reset_all"()

    void cpp_device_print_all_gpu_specification "merlin::device::print_all_gpu_specification" ()
    bint cpp_device_test_all_gpu "merlin::device::test_all_gpu" ()
