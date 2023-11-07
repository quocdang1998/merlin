# Copyright 2023 quocdang1998

from libc.stdint cimport uint64_t
from libcpp.string cimport string

from merlin.vector cimport CppVector, CppFloatvec, CppIntvec

cdef extern from "merlin/splint/cartesian_grid.hpp":

    cdef cppclass CppCartesianGrid "merlin::splint::CartesianGrid":
        # Default constructors
        CppCartesianGrid()
        CppCartesianGrid(const CppFloatvec & grid_nodes, const CppIntvec & shape) except+

        # Copy constructor and copy assignment
        CppCartesianGrid(const CppCartesianGrid & src)
        CppCartesianGrid & operator=(const CppCartesianGrid & src)

        # Get members and attributes
        const CppFloatvec grid_vector(uint64_t i_dim)
        uint64_t ndim()
        const CppIntvec & shape()
        uint64_t size()
        uint64_t num_nodes()

        # Slicing operator
        CppFloatvec operator[](uint64_t index)
        CppFloatvec operator[](const CppIntvec & index)

        string str()

