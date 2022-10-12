# Copyright 2022 quocdang1998

from gpu_query cimport *

def get_current_gpu():
    return Cpp_device_get_current_gpu()

cdef class Device:

    cdef Cpp_device_Device * core

    def __init__(self, int id = -1):
        self.core = new Cpp_device_Device(id)

    @classmethod
    def get_num_gpu(self):
        return cpp_device_Device_get_num_gpu()

    def print_specification(self):
        self.core.print_specification()

    def test_gpu(self):
        return self.core.test_gpu()

    @classmethod
    def reset_all(self):
        cpp_device_Device_reset_all()

def print_all_gpu_specification():
    cpp_device_print_all_gpu_specification()

def test_all_gpu():
    return cpp_device_test_all_gpu()
