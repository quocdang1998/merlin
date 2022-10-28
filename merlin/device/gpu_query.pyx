# Copyright 2022 quocdang1998
from cpython.unicode cimport PyUnicode_FromString
from libc.stdint cimport uint64_t, UINT64_MAX
from libcpp.string cimport string

cdef extern from "merlin/device/gpu_query.hpp":
    int Cpp_device_get_current_gpu "merlin::device::get_current_gpu" ()

    cpdef enum class DeviceLimit "merlin::device::Device::Limit":
        StackSize,
        PrintfSize,
        HeapSize,
        SyncDepth,
        LaunchPendingCount

    cdef cppclass Cpp_device_Device "merlin::device::Device":
        Cpp_device_Device(int id)
        Cpp_device_Device(const Cpp_device_Device & src)
        Cpp_device_Device & operator=(const Cpp_device_Device & src)
        void print_specification()
        bint test_gpu()
        string repr()
    int cpp_device_Device_get_num_gpu "merlin::device::Device::get_num_gpu" ()
    void cpp_device_Device_reset_all "merlin::device::Device::reset_all" ()
    uint64_t cpp_device_Device_limit "merlin::device::Device::limit" (DeviceLimit limit, uint64_t size)

    void cpp_device_print_all_gpu_specification "merlin::device::print_all_gpu_specification" ()
    bint cpp_device_test_all_gpu "merlin::device::test_all_gpu" ()

def get_current_gpu():
    return Cpp_device_get_current_gpu()

cdef class Device:
    """Represent a GPU."""

    cdef Cpp_device_Device * core

    def __init__(self, int id = -1):
        """__init__(self, id = -1)
        Initializer.

        Parameters
        ----------
        id: int
            ID of GPU (from 0 to number of GPU - 1). If ``id == -1``, the GPU
            currently used is assigned.
        """
        self.core = new Cpp_device_Device(id)

    def __repr__(self):
        return PyUnicode_FromString(self.core.repr().c_str())

    @classmethod
    def get_num_gpu(self):
        """get_num_gpu(self)
        Get number of GPU detected.

        Returns
        -------
        ``int``
            Number of GPU detected.
        """
        return cpp_device_Device_get_num_gpu()

    def print_specification(self):
        """print_specification(self)
        Print specification of the GPU.
        """
        self.core.print_specification()

    def test_gpu(self):
        """test_gpu(self)
        Perform a simple addition of 2 integers on GPU and compare with the
        result on CPU to ensure the proper functionality of GPU and CUDA.

        Returns
        -------
        ``bool``
            ``False`` if the test has failed.
        """
        return self.core.test_gpu()

    @classmethod
    def limit(self, DeviceLimit limit, uint64_t size = UINT64_MAX):
        return cpp_device_Device_limit(limit, size)

    @classmethod
    def reset_all(self):
        """reset_all(self)
        Reset all GPU (halt kernels launched and free allocated memory).
        """
        cpp_device_Device_reset_all()

def print_all_gpu_specification():
    """
    Print specification of all detected GPUs.
    """
    cpp_device_print_all_gpu_specification()

def test_all_gpu():
    """
    Perform the test on all detected GPU.
    """
    return cpp_device_test_all_gpu()
