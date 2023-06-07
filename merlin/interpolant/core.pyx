# Copyright 2023 quocdang1998

from cython.operator cimport dereference
from cpython.unicode cimport PyUnicode_FromString
from libc.stdint cimport uintptr_t
from libcpp.utility cimport move

from merlin.vector cimport CppVector, CppIntvec, CppFloatvec, intvec_from_iteratable, list_from_intvec

from merlin.array.array cimport CppArray
from merlin.interpolant.cartesian_grid cimport CppCartesianGrid
from merlin.interpolant.interpolant cimport CppPolynomialInterpolant

from merlin.array import Array

include "enum_method.pxd"
include "cartesian_grid.pyx"
include "interpolant.pyx"
