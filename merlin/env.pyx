# Copyright 2022 quocdang1998

from cython.operator cimport dereference
from libc.stdint cimport uint64_t, UINT64_MAX

from merlin.env cimport *

cdef class Environment:
    """Execution environment of package."""
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

    @classmethod
    def parallel_chunk(self, uint64_t n_chunks = UINT64_MAX):
        """parallel_chunk(self, n_chunks=UINT64_MAX)
        Get or set the minimum size of loops so CPU parallelization is applied.

        Default value is ``96``.
        """
        global CppEnvironment_parallel_chunk
        if n_chunks != UINT64_MAX:
            CppEnvironment_parallel_chunk = n_chunks
        return CppEnvironment_parallel_chunk

    def __repr__(self):
        return "<Merlin execution Environment>"

    def __dealloc__(self):
        del self.core
