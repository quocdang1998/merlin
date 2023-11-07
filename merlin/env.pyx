# Copyright 2022 quocdang1998

from cython.operator cimport dereference
from libc.stdint cimport uint64_t, UINT64_MAX

from merlin.env cimport *

cdef class Environment:
    """
    Execution environment of the ``merlin`` library.
    
    Wrapper of the class :cpp:class:`merlin::Environment`.
    """

    cdef CppEnvironment * core

    def __init__(self):
        self.core = new CppEnvironment()

    @classmethod
    def is_initialized(self):
        """is_initialized(self)
        Check if environment is initialized at the beginning of the program.
        """
        return CppEnvironment_is_initialized

    @classmethod
    def num_instances(self):
        """num_instances(self)
        Get number of isntances initialized.
        """
        return CppEnvironment_num_instances.load()

    def __repr__(self):
        return "<Merlin execution Environment>"

    def __dealloc__(self):
        del self.core
