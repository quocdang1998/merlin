# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.string cimport string

from merlin.vector cimport CppVector, CppIntvec

cdef extern from "merlin/array/nddata.hpp":

    cdef cppclass CppNdData "merlin::array::NdData":
        CppNdData()
        CppNdData(double * data, const CppIntvec & shape, const CppIntvec & strides)
        CppNdData(const CppIntvec & shape)

        CppNdData(const CppNdData & src)
        CppNdData & operator=(const CppNdData & src)

        double * data()
        uint64_t ndim()
        const CppIntvec & shape()
        const CppIntvec & strides()

        uint64_t size()
        double get(const CppIntvec & index) except +
        double get(uint64_t index) except +
        void set(const CppIntvec & index, double value) except +
        void set(uint64_t index, double value) except +

        string str() except +
