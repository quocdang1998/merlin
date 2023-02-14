# Copyright 2022 quocdang1998

cdef extern from "merlin/env.hpp":

    cdef cppclass CppEnvironment "merlin::Environment":
        CppEnvironment() except +

        void set_inited(bint value) except +
        void print_inited() except +

cdef class Environment:
    cdef CppEnvironment * core

    def __init__(self):
        self.core = new CppEnvironment()

    def set_inited(self, bint value):
        self.core.set_inited(value)

    def print_inited(self):
        self.core.print_inited()

    def __dealloc__(self):
        del self.core
