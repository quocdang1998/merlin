# Copyright 2022 quocdang1998

cdef class Context:
    """CUDA context attached to a CPU process."""

    cdef Cpp_Context * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Parameters
        ----------
        gpu: merlin.device.Device
            GPU.
        flag: merlin.device.ContextFlags
        """
        cdef Device gpu
        cdef ContextFlags flag

        if not kwargs:
            self.core = new Cpp_Context()
        elif kwargs.get("gpu") is not None:
            gpu = kwargs.pop("gpu")
            if not isinstance(gpu, Device):
                raise ValueError("Expected argument \"gpu\" has type merlin.device.Device")
            if kwargs.get("flag") is not None:
                flag = kwargs.pop("flag")
                self.core = new Cpp_Context(dereference(<Cpp_Device*>(gpu.core)), flag)
            else:
                self.core = new Cpp_Context(dereference(<Cpp_Device*>(gpu.core)), ContextFlags.AutoSchedule)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    def get_gpu(self):
        result = Device()
        cdef Cpp_Device * c_result = new Cpp_Device(self.core.get_gpu())
        result.c_assign(c_result)
        return result

    def is_attached(self):
        return self.core.is_attached()

    def __dealloc__(self):
        del self.core
