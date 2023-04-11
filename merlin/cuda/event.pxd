# Copyright 2022 quocdang1998

from libc.stdint cimport uintptr_t
from libcpp.string cimport string

from merlin.cuda.device cimport CppDevice
from merlin.cuda.context cimport CppContext
from merlin.cuda.enum_types cimport EventCategory

cdef extern from "merlin/cuda/event.hpp":

    cdef cppclass CppEvent "merlin::cuda::Event":
        CppEvent() except +
        CppEvent(unsigned int category) except +

        # CppEvent(const CppEvent & src) except +

        uintptr_t get_event_ptr()
        unsigned int category()
        const CppContext & get_context() except +
        const CppDevice & get_gpu()

        bint is_complete() except +
        void check_cuda_context() except +
        void synchronize() except +

        string str() except +

    float operator-(const CppEvent & ev_1, const CppEvent & ev_2)
