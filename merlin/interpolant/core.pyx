# Copyright 2023 quocdang1998

from cpython.unicode cimport PyUnicode_FromString
from libc.stdint cimport uintptr_t
from libcpp.utility cimport move

from merlin.vector cimport CppVector, CppIntvec, intvec_from_iteratable, list_from_intvec

from merlin.interpolant.cartesian_grid cimport CppCartesianGrid

include "cartesian_grid.pyx"
