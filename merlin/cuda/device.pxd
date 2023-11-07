# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.string cimport string

from merlin.cuda.enum_types cimport DeviceLimit

cdef extern from "merlin/cuda/device.hpp":

    cdef cppclass CppDevice "merlin::cuda::Device":
        CppDevice() except +
        CppDevice(int id) except +

        CppDevice(const CppDevice & src) except +

        void print_specification() except +
        bint test_gpu() except +

        void set_as_current() except +
        string str() except +
        int & id() except +

        @staticmethod
        CppDevice get_current_gpu() except +
        @staticmethod
        uint64_t get_num_gpu() except +
        @staticmethod
        uint64_t limit(DeviceLimit limit, uint64_t size) except +
        @staticmethod
        void reset_all() except+

    bint operator==(const CppDevice & left, const CppDevice & right) except +
    bint operator!=(const CppDevice & left, const CppDevice & right) except +

    void cpp_print_gpus_spec "merlin::cuda::print_gpus_spec" () except +
    bint cpp_test_all_gpu "merlin::cuda::test_all_gpu" () except +
