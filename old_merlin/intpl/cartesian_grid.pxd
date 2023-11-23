# Copyright 2023 quocdang1998

from libcpp.string cimport string

from merlin.vector cimport CppVector, CppIntvec

cdef extern from "merlin/intpl/cartesian_grid.hpp":

    cdef cppclass CppCartesianGrid "merlin::intpl::CartesianGrid":
        CppCartesianGrid()
        CppCartesianGrid(const CppVector[CppVector[double]] & grid_vectors)

        CppCartesianGrid(const CppCartesianGrid & src)
        CppCartesianGrid & operator=(const CppCartesianGrid & src)

        CppIntvec get_grid_shape()

        string str()

