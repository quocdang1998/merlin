# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t

from merlin.vector cimport CppVector, CppIntvec

from merlin.array.nddata cimport CppNdData
from merlin.array.slice cimport CppSlice

cdef extern from "merlin/array/array.hpp":

    cdef cppclass CppArray "merlin::array::Array" (CppNdData):
        CppArray()
        CppArray(double value)
        CppArray(double * data, uint64_t ndim, const uint64_t * shape, const uint64_t * strides, bint copy)
        CppArray(const CppIntvec & shape)
        CppArray(const CppArray & whole, const CppVector[CppSlice] & slices)

        CppArray(const CppArray & src)
        CppArray & operator=(const CppArray & src)

        double & operator[](const CppIntvec & index)
