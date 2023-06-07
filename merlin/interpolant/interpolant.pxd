# Copyright 2023 quocdang1998

from merlin.array.array cimport CppArray
from merlin.array.nddata cimport CppNdData
from merlin.interpolant.cartesian_grid cimport CppCartesianGrid
from merlin.interpolant.enum_method cimport Method
from merlin.vector cimport CppFloatvec

cdef extern from "merlin/interpolant/interpolant.hpp":

    cdef cppclass CppPolynomialInterpolant "merlin::interpolant::PolynomialInterpolant":
        CppPolynomialInterpolant() except +
        CppPolynomialInterpolant(const CppCartesianGrid & grid, const CppArray & values, Method method) except +

        CppPolynomialInterpolant(const CppPolynomialInterpolant & src)
        CppPolynomialInterpolant & operator=(const CppPolynomialInterpolant & src)

        CppCartesianGrid & get_grid() except +
        CppNdData & get_coeff() except +

        double operator()(const CppFloatvec & point)
