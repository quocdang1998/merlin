# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t

cdef extern from "merlin/env.hpp":

    cdef cppclass CppEnvironment "merlin::Environment":
        CppEnvironment() except +

    cdef bint CppEnvironment_is_initialized "merlin::Environment::is_initialized"

    cdef uint64_t CppEnvironment_cpu_mem_limit "merlin::Environment::cpu_mem_limit"

    cdef int CppEnvironment_default_gpu "merlin::Environment::default_gpu"
