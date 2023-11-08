# Copyright 2023 quocdang1998

from cython.operator cimport dereference
from cpython.unicode cimport PyUnicode_FromString
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.utility cimport move

from merlin.vector cimport CppVector, CppFloatvec, CppIntvec, list_from_intvec

from merlin.splint.cartesian_grid cimport CppCartesianGrid
from merlin.splint.interpolator cimport CppInterpolator

include "enum_method.pxd"
include "cartesian_grid.pyx"
include "interpolator.pyx"
