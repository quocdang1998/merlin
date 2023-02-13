# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.string cimport string

cdef extern from "merlin/utils.hpp":

    string cpp_get_current_process_id() except +
    string get_time() except +

    uint64_t cpp_inner_prod(const CppIntvec & v1, const CppIntvec & v2) except +
    uint64_t cpp_ndim_to_contiguous_idx(const CppIntvec & index, const CppIntvec & shape) except +
    CppIntvec cpp_contiguous_to_ndim_idx(uint64_t index, const CppIntvec & shape) except +

