# Copyright 2022 quocdang1998

from cython.operator cimport dereference
from libc.stdint cimport uint64_t, UINT64_MAX

from merlin.env cimport *

cdef class Environment:
    cdef CppEnvironment * core

    def __init__(self):
        self.core = new CppEnvironment()

    @classmethod
    def is_initialized(self):
        return CppEnvironment_is_initialized

    @classmethod
    def cpu_mem_limit(self, uint64_t limit = UINT64_MAX):
        global CppEnvironment_cpu_mem_limit
        if limit != UINT64_MAX:
            CppEnvironment_cpu_mem_limit = limit
        return CppEnvironment_cpu_mem_limit

    @classmethod
    def default_gpu(self):
        return CppEnvironment_default_gpu

    def __dealloc__(self):
        del self.core
