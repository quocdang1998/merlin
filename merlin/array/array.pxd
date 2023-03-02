# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t

from merlin.cuda.stream cimport CppStream
from merlin.vector cimport CppVector, CppIntvec

from merlin.array.nddata cimport CppNdData
from merlin.array.parcel cimport CppParcel
from merlin.array.slice cimport CppSlice
from merlin.array.stock cimport CppStock

cdef extern from "merlin/array/array.hpp":

    cdef cppclass CppArray "merlin::array::Array" (CppNdData):
        CppArray()
        CppArray(double value)
        CppArray(double * data, const CppIntvec & shape, const CppIntvec & strides, bint copy)
        CppArray(const CppIntvec & shape)
        CppArray(const CppArray & whole, const CppVector[CppSlice] & slices)

        CppArray(const CppArray & src)
        CppArray & operator=(const CppArray & src)

        double & operator[](const CppIntvec & index)

        void clone_data_from_gpu(const CppParcel & src, const CppStream & stream)
        void extract_data_from_file(const CppStock & src)
