# Copyright 2023 quocdang1998

from libc.stdint cimport uint64_t

from merlin.array.array cimport CppArray
from merlin.splint.cartesian_grid cimport CppCartesianGrid
from merlin.splint.enum_method cimport Method
from merlin.vector cimport CppVector

cdef extern from "merlin/splint/interpolator.hpp":

    cdef cppclass CppInterpolator "merlin::splint::Interpolator":
        # Default constructors
        CppInterpolator()
        CppInterpolator(const CppCartesianGrid & grid, CppArray & values, const CppVector[Method] & method,
                        uint64_t n_threads) except+



