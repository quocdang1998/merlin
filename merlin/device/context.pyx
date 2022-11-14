# Copyright 2022 quocdang1998

cdef class Context:
    """CUDA context attached to a CPU process."""

    cdef CppContext * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Parameters
        ----------
        gpu: merlin.device.Device
            GPU.
        flag: merlin.device.ContextFlags
            Setting flag.
        """
        cdef Device gpu
        cdef ContextFlags flag

        if not kwargs:
            self.core = new CppContext()
        elif kwargs.get("gpu") is not None:
            gpu = kwargs.pop("gpu")
            if kwargs.get("flag") is not None:
                flag = kwargs.pop("flag")
                self.core = new CppContext(dereference(<CppDevice*>(gpu.core)), flag)
            else:
                self.core = new CppContext(dereference(<CppDevice*>(gpu.core)), ContextFlags.AutoSchedule)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    def assign(self, uintptr_t ptr):
        """assign(self, ptr)
        Assign pointer to a C++ object to the Python class wrapper.

        Parameters
        ----------
        ptr: int
            Pointer to C++ object to assign to current object in form of an unsigned integer.
        """
        del self.core
        self.core = <CppContext *>(ptr)

    cdef c_assign(self, CppContext * new_core):
        del self.core
        self.core = new_core

    cpdef uintptr_t pointer(self):
        """pointer(self)
        Return pointer to C++ object wrapped by the class instance.

        Returns
        -------
        ``int``
            Pointer to C++ object wrapped by the object instance in form of an unsigned integer.
        """
        return <uintptr_t>(self.core)

    def get_gpu(self):
        """get_gpu(self)
        Get GPU assigned to the context.
        """
        result = Device()
        cdef CppDevice * c_result = new CppDevice(self.core.get_gpu())
        result.c_assign(c_result)
        return result

    def is_attached(self):
        """is_attached(self)
        Check if the context is attached to CPU process.
        """
        return self.core.is_attached()

    def push_current(self):
        """push_current(self)
        Push the context to the context stack.
        """
        self.core.push_current()

    def pop_current(self):
        """pop_current(self)
        Pop the context out of the context stack.
        """
        result = Context()
        cdef CppContext * c_result = &self.core.pop_current()
        result.c_assign(c_result)
        return result

    def is_current(self):
        """is_current(self)
        Check if the context is the top of the context stack.
        """
        return self.core.is_current()

    def set_current(self):
        """set_current(self)
        Set the context as the top context of the stack.
        """
        self.core.set_current()

    @classmethod
    def get_primary_context(self, Device gpu):
        result = Context()
        cdef CppContext * c_result = &CppContext.get_primary_context(dereference(gpu.core))
        result.c_assign(c_result)
        return result

    @classmethod
    def get_primary_ctx_state(self, Device gpu):
        cdef pair[bint, ContextFlags] c_result = CppContext.get_primary_ctx_state(dereference(gpu.core))
        return (c_result.first, c_result.second)

    def __dealloc__(self):
        del self.core
