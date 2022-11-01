# Copyright 2022 quocdang1998

cdef class Context:
    """CUDA context attached to a CPU process."""

    cdef Cpp_device_Context * core

    def __init__(Context self, **kwargs):
        self.core = new Cpp_device_Context()

    def get_gpu(self):
        return Device(self.core.get_gpu().id())

    def __dealloc__(self):
        del self.core
