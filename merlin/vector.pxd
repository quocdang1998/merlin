# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.string cimport string

cimport numpy as np

cdef extern from "merlin/vector.hpp":

    cdef cppclass CppVector "merlin::Vector" [T, Convertable=*]:
        CppVector()
        CppVector(uint64_t size, const T & value)
        CppVector(const Convertable * ptr_first, const Convertable * ptr_last)
        CppVector(const Convertable * ptr_src, uint64_t size)

        CppVector(const CppVector[T] & src)
        CppVector[T] & operator=(const CppVector[T] & src)

        void assign(T * ptr_src, uint64_t size)
        void assign(T * ptr_first, T * ptr_last)

        T * & data()
        uint64_t & size()
        T * begin()
        T * end()
        T & operator[](uint64_t index)

        uint64_t malloc_size()
        void * copy_to_gpu(CppVector[T] * gpu_ptr, void * data_ptr) except +
        void copy_from_device(CppVector[T] * gpu_ptr) except +

        string str(const char * sep = " ")

    bint operator==[T](const CppVector[T] & vec_1, const CppVector[T] & vec_2)
    bint operator!=[T](const CppVector[T] & vec_1, const CppVector[T] & vec_2)

    ctypedef CppVector[uint64_t, np.npy_intp] CppIntvec "merlin::intvec"


cdef inline CppIntvec intvec_from_tuple(tuple values):
    cdef CppIntvec result = CppIntvec(len(values), 0)
    for i in range(len(values)):
        if values[i] < 0:
            raise ValueError("Expected non-negative tuple of integers.")
        result[i] = <uint64_t>(values[i])
    return result

cdef inline tuple tuple_from_intvec(const CppIntvec & values):
    cdef list result = []
    for i in range(values.size()):
        result.append(values[i])
    return tuple(result)
