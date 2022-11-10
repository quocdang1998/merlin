# Copyright 2022 quocdang1998

cdef class Device:
    """Represent a GPU."""

    cdef Cpp_Device * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Parameters
        ----------
        id: int
            ID of GPU (from 0 to number of GPU - 1).
        """
        cdef int id

        if not kwargs:
            self.core = new Cpp_Device()
        elif kwargs.get("id") is not None:
            id = kwargs.pop("id")
            self.core = new Cpp_Device(id)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    def __repr__(self):
        return PyUnicode_FromString(self.core.repr().c_str())

    def assign(self, uintptr_t ptr):
        """assign(self, ptr)
        Assign pointer to a C++ object to the Python class wrapper.
        """
        del self.core
        self.core = <Cpp_Device *>(ptr)

    cdef c_assign(self, Cpp_Device * new_core):
        del self.core
        self.core = new_core

    cpdef uintptr_t pointer(self):
        """pointer(self)
        Return pointer to C++ object wrapped by the class instance.
        """
        return <uintptr_t>(self.core)

    @classmethod
    def get_num_gpu(self):
        """get_num_gpu(self)
        Get number of GPU detected.

        Returns
        -------
        ``int``
            Number of GPU detected.
        """
        return cpp_Device_get_num_gpu()

    @classmethod
    def get_current_gpu(self):
        """get_current_gpu(self)
        Get instance pointing to current GPU.

        Returns
        -------
        ``merlin.device.Device``
            Current GPU.
        """
        result = Device()
        cdef Cpp_Device * c_result = new Cpp_Device(cpp_Device_get_current_gpu())
        result.c_assign(c_result)
        return result

    def print_specification(self):
        """print_specification(self)
        Print specification of the GPU.
        """
        self.core.print_specification()

    def test_gpu(self):
        """test_gpu(self)
        Perform a simple addition of 2 integers on GPU and compare with the
        result on CPU to ensure the proper functionality of GPU and CUDA.

        Returns
        -------
        ``bool``
            ``False`` if the test has failed.
        """
        return self.core.test_gpu()

    def set_as_current(self):
        """set_as_current(self)
        Set as current GPU.
        """
        self.core.set_as_current()

    @classmethod
    def limit(self, DeviceLimit limit, uint64_t size = UINT64_MAX):
        """limit(limit, size = UINT64_MAX)
        Get (in case of the argument ``size`` is not provided) or set the limit of GPU.

        Parameters
        ----------
        limit: DeviceLimit
            The limit to get or set of the GPU.
        size: int
            If provided, set the limit of the GPU to be the value of the argument.
        """
        return cpp_Device_limit(limit, size)

    @classmethod
    def reset_all(self):
        """reset_all(self)
        Reset all GPU (halt kernels launched and free allocated memory).
        """
        cpp_Device_reset_all()

    def __eq__(Device left, Device right):
        dereference(left.core) == dereference(right.core)

    def __ne__(Device left, Device right):
        dereference(left.core) != dereference(right.core)

    def __dealloc__(self):
        del self.core

def print_all_gpu_specification():
    """
    Print specification of all detected GPUs.
    """
    cpp_print_all_gpu_specification()

def test_all_gpu():
    """
    Perform the test on all detected GPU.
    """
    return cpp_test_all_gpu()
