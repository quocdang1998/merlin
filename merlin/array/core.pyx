# Copyright 2022 quocdang1998

from cpython.ref cimport Py_INCREF, Py_DECREF
from cpython.unicode cimport PyUnicode_FromString
from cython.operator cimport dereference
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.string cimport string

cimport numpy as np

from merlin.vector cimport CppVector, CppIntvec, intvec_from_iteratable, tuple_from_intvec
from merlin.cuda.device cimport CppDevice
from merlin.cuda.stream cimport CppStream

from merlin.array.array cimport CppArray
from merlin.array.nddata cimport CppNdData
from merlin.array.parcel cimport CppParcel
from merlin.array.slice cimport CppSlice
from merlin.array.stock cimport CppStock

from merlin.cuda import Device, Stream

include "nddata.pyx"
include "array.pyx"
include "parcel.pyx"
include "stock.pyx"
