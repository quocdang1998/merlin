# Copyright 2022 quocdang1998

cdef extern from "merlin/cuda/stream.hpp":

    cpdef enum class StreamSetting "merlin::cuda::Stream::Setting":
        """Parameter controlling the behavior of the stream.

        *Values*

         - ``Default``: Default stream creation flag (synchonized with the null stream).
         - ``NonBlocking``: Works may run concurrently with null stream.
        """
        Default,
        NonBlocking

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
