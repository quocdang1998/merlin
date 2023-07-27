# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.atomic cimport atomic

cdef extern from "merlin/env.hpp":

    cdef cppclass CppEnvironment "merlin::Environment":
        CppEnvironment() except +

    cdef bint CppEnvironment_is_initialized "merlin::Environment::is_initialized"
    cdef atomic[unsigned int] CppEnvironment_num_instances "merlin::Environment::num_instances"

    cdef uint64_t CppEnvironment_parallel_chunk "merlin::Environment::parallel_chunk"

    cdef int CppEnvironment_default_gpu "merlin::Environment::default_gpu"
