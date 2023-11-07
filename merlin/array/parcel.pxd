# Copyright 2022 quocdang1998

from libc.stdint cimport uint64_t

from merlin.vector cimport CppVector, CppIntvec
from merlin.cuda.device cimport CppDevice
from merlin.cuda.stream cimport CppStream

from merlin.array.array cimport CppArray
from merlin.array.nddata cimport CppNdData

cdef extern from "merlin/array/parcel.hpp":

    cdef cppclass CppParcel "merlin::array::Parcel" (CppNdData):
        CppParcel()
        CppParcel(const CppIntvec & shape) except +

        CppParcel(const CppParcel & src) except +
        CppParcel & operator=(const CppParcel & src) except +

        const CppDevice & device()

        void transfer_data_to_gpu(const CppArray & cpu_array, const CppStream & stream) except +
        uint64_t malloc_size()
