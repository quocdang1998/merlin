# Copyright 2022 quocdang1998

cdef class Event:
    """Wrapper class for CUDA event."""

    cdef CppEvent * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Parameters
        ----------
        category: merlin.cuda.EventCategory
            Setting flag of the event.
        """
        cdef unsigned int category

        if not kwargs:
            self.core = new CppEvent()
        elif kwargs.get("category") is not None:
            category = kwargs.pop("category")
            self.core = new CppEvent(category)

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
        self.core = <CppEvent *>(ptr)

    cdef c_assign(self, CppEvent * new_core):
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

    def category(self):
        """category(self)
        Return the setting flag of the event.
        """
        return self.core.category()

    def get_gpu(self):
        """get_gpu(self)
        Return the GPU binded to the event.
        """
        result = Device()
        cdef CppDevice * c_result = new CppDevice(move(self.core.get_gpu()))
        result.c_assign(c_result)
        return result

    def is_complete(self):
        """is_complete(self)
        Query the status of all work currently captured by event.

        Returns
        -------
        ``bool``
            ``True`` if all captured work has been completed.
        """
        return self.core.is_complete()

    def check_cuda_context(self):
        """check_cuda_context(self)
        Check if the current CUDA context and active GPU is valid for the event.
        """
        self.core.check_cuda_context()

    def synchronize(self):
        """synchonize(self)
        Pause the CPU process until the event occurs.
        """
        self.core.synchronize()

    def __sub__(Event left, Event right):
        """__sub__(Event left, Event right)
        Calculate elapsed time (in millisecond) between 2 events.
        """
        return dereference(left.core) - dereference(right.core)

    def __dealloc__(self):
        del self.core
