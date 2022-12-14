# Copyright 2022 quocdang1998

cdef class Context:
    """CUDA context attached to a CPU process."""

    cdef CppContext * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Parameters
        ----------
        gpu: merlin.cuda.Device
            GPU.
        flag: merlin.cuda.ContextFlags
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

    def __repr__(self):
        return PyUnicode_FromString(self.core.str().c_str())

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

    def push_current(self):
        """push_current(self)
        Push the context to the context stack.
        """
        self.core.push_current()

    def pop_current(self):
        """pop_current(self)
        Pop the context out of the context stack.
        """
        self.core.pop_current()
        return self

    def is_current(self):
        """is_current(self)
        Check if the context is the top of the context stack.
        """
        return self.core.is_current()

    @classmethod
    def get_current(self):
        """get_current(self)
        Get the current context.
        """
        result = Context()
        cdef CppContext * c_result = new CppContext(move(CppContext.get_current()))
        result.c_assign(c_result)
        return result

    @classmethod
    def get_gpu_of_current_context(self):
        """get_gpu_of_current_context(self)
        Get the GPU currently associated to the current context.
        """
        cdef CppDevice c_device = CppContext.get_gpu_of_current_context()
        return Device(id=c_device.id())

    @classmethod
    def get_flag_of_current_context(self):
        """get_flag_of_current_context(self)
        Get the setting flag of the current context.
        """
        cdef ContextFlags flag = CppContext.get_flag_of_current_context()
        return ContextFlags(flag)

    def __eq__(Context left, Context right):
        return dereference(left.core) == dereference(right.core)

    def __ne__(Context left, Context right):
        return dereference(left.core) == dereference(right.core)

    def __dealloc__(self):
        del self.core

def create_primary_context(Device gpu, ContextFlags flag):
    """
    Retain the primary context associated to a GPU.

    Primary contexts are contexts shared with the CUDA driver API. There is a correspondance one-to-one between primary
    contexts and GPU.
    If the primary context of the GPU has been retained, the function only change the flag of the context.
    """
    result = Context()
    cdef CppContext * c_result = new CppContext(cpp_create_primary_context(dereference(gpu.core), flag))
    result.c_assign(c_result)
    return result
