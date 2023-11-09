# Copyright 2023 quocdang1998

from libc.stdint cimport uint64_t

from merlin.array.array cimport CppArray
from merlin.array.parcel cimport CppParcel
from merlin.cuda.stream cimport CppStream
from merlin.splint.cartesian_grid cimport CppCartesianGrid
from merlin.splint.enum_method cimport Method
from merlin.vector cimport CppVector

cdef extern from "merlin/splint/interpolator.hpp":

    cdef cppclass CppInterpolator "merlin::splint::Interpolator":
        # Default constructors
        CppInterpolator()
        CppInterpolator(const CppCartesianGrid & grid, CppArray & values, CppVector[Method] & method,
                        uint64_t n_threads) except+
        CppInterpolator(const CppCartesianGrid & grid, CppParcel & values, CppVector[Method] & method,
                        const CppStream & stream, uint64_t n_threads)
