# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.string cimport string

from merlin.vector cimport CppVector, CppIntvec

from merlin.array.array cimport CppArray
from merlin.array.nddata cimport CppNdData

cdef extern from "merlin/array/stock.hpp":

    cdef cppclass CppStock "merlin::array::Stock" (CppNdData):
        CppStock()
        CppStock(const string & filename, const CppIntvec & shape, uint64_t offset, bint thread_safe)
        CppStock(const string & filename, uint64_t offset, bint thread_safe)

        CppStock(const CppStock & src)
        CppStock & operator=(const CppStock & src)

        void record_data_to_file(const CppArray & src)
