# Copyright 2022 quocdang1998

from libc.stdint cimport uintptr_t
from libcpp.string cimport string

from merlin.cuda.device cimport CppDevice
from merlin.cuda.context cimport CppContext

cdef extern from "merlin/cuda/event.hpp":

    cpdef enum class EventCategory "merlin::cuda::Event::Category":
        """Parameter controlling the behavior of the stream.

        *Values*

         - ``Default``: Default event.
         - ``BlockingSync``: Event meant to be synchronize with CPU (process on CPU blockled until the event occurs).
         - ``DisableTiming``: Event not recording time data.
         - ``EventInterprocess``: Event might be used in an interprocess communication.
        """
        Default,
        BlockingSync,
        DisableTiming,
        EventInterprocess

    cdef cppclass CppEvent "merlin::cuda::Event":
        CppEvent() except +
        CppEvent(EventCategory category) except +
        CppEvent(const CppEvent & src) except +
        uintptr_t get_event_ptr()
        EventCategory category()
        const CppContext & get_context() except +
        const CppDevice & get_gpu()
        bint is_complete() except +
        void check_cuda_context() except +
        void synchronize() except +
        string str() except +

    float operator-(const CppEvent & ev_1, const CppEvent & ev_2)
