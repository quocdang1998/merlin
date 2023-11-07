# Copyright 2022 quocdang1998

cdef class Stream:
    """Wrapper class for CUDA stream."""

    cdef CppStream * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Parameters
        ----------
        setting: merlin.cuda.StreamSetting
            Mode of the stream.
        priority: int
            Priority of the stream. The smaller value, the higher priority.
        """
        cdef StreamSetting setting
        cdef int priority

        if not kwargs:
            self.core = new CppStream()
        elif kwargs.get("setting") is not None:
            setting = kwargs.pop("setting")
            if kwargs.get("priority") is not None:
                priority = kwargs.pop("priority")
                self.core = new CppStream(setting, priority)
            else:
                self.core = new CppStream(setting, 0)

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
        self.core = <CppStream *>(ptr)

    cdef c_assign(self, CppStream * new_core):
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

    def setting(self):
        """setting(self)
        Return the setting flag of the stream.
        """
        return StreamSetting(self.core.setting())

    def priority(self):
        """priority(self)
        Return the priority of the stream.
        """
        return self.core.priority()

    def get_gpu(self):
        """get_gpu(self)
        Return the GPU binded to the stream.
        """
        result = Device()
        cdef CppDevice * c_result = new CppDevice(move(self.core.get_gpu()))
        result.c_assign(c_result)
        return result

    def is_complete(self):
        """is_complete(self)
        Query for completion status.

        Returns
        -------
        ``bool``
            ``True`` if all operations in the stream have completed.
        """
        return self.core.is_complete()

    def check_cuda_context(self):
        """check_cuda_context(self)
        Check if the current CUDA context and active GPU is valid for the stream.
        """
        self.core.check_cuda_context()

    def record_event(self, Event event):
        """record_event(self, event)
        Record an event to CUDA stream.
        """
        self.core.record_event(dereference(event.core))

    def synchronize(self):
        """synchronize(self)
        Pause the CPU process until all operations on the stream has finished.
        """
        self.core.synchronize()

    def __dealloc__(self):
        del self.core
