# Copyright 2022 quocdang1998

from libc.stdint cimport uintptr_t
from libcpp.string cimport string

from merlin.cuda.device cimport CppDevice
from merlin.cuda.context cimport CppContext
from merlin.cuda.event cimport CppEvent
from merlin.cuda.enum_types cimport StreamSetting

cdef extern from "merlin/cuda/stream.hpp":

    cdef cppclass CppStream "merlin::cuda::Stream":
        CppStream() except +
        CppStream(StreamSetting setting, int priority) except +

        CppStream(const CppStream & src) except +

        uintptr_t get_stream_ptr()
        StreamSetting setting() except +
        int priority() except +
        CppContext get_context() except +
        const CppDevice & get_gpu()

        bint is_complete() except +
        void check_cuda_context() except +
        void synchronize() except +

        string str() except +

    cdef void cpp_record_event "merlin::cuda::record_event" (const CppEvent & event, const CppStream & stream)
