# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t, UINT64_MAX
from libcpp.string cimport string

cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass array3 "std::array<std::uint64_t, 3>":
        array3() except+
        uint64_t & operator[](size_t)

cdef extern from "merlin/array/slice.hpp":

    cdef cppclass CppSlice "merlin::array::Slice":
        CppSlice(uint64_t start = 0, uint64_t stop = UINT64_MAX, uint64_t step = 1) except +

        CppSlice(const CppSlice & src)
        CppSlice & operator=(const CppSlice & src)

        uint64_t & start()
        uint64_t & stop()
        uint64_t & step()

        array3 slice_on(uint64_t shape, uint64_t stride) except +
        uint64_t get_index_in_whole_array(uint64_t index_sliced_array)
        bint in_slice(uint64_t index)

        string str()

