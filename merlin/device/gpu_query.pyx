# Copyright 2022 quocdang1998

def get_current_gpu():
    return Cpp_device_get_current_gpu()

cdef class Device:
    """Represent a GPU."""

    cdef Cpp_device_Device * core

    def __init__(self, int id = -1):
        """__init__(self, id = -1)
        Initializer.

        Parameters
        ----------
        id: int
            ID of GPU (from 0 to number of GPU - 1). If ``id == -1``, the GPU
            currently used is assigned.
        """
        self.core = new Cpp_device_Device(id)

    def __repr__(self):
        return PyUnicode_FromString(self.core.repr().c_str())

    @classmethod
    def get_num_gpu(self):
        """get_num_gpu(self)
        Get number of GPU detected.

        Returns
        -------
        ``int``
            Number of GPU detected.
        """
        return cpp_device_Device_get_num_gpu()

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
        return cpp_device_Device_limit(limit, size)

    @classmethod
    def reset_all(self):
        """reset_all(self)
        Reset all GPU (halt kernels launched and free allocated memory).
        """
        cpp_device_Device_reset_all()
    
    def __dealloc__(self):
        del self.core

def print_all_gpu_specification():
    """
    Print specification of all detected GPUs.
    """
    cpp_device_print_all_gpu_specification()

def test_all_gpu():
    """
    Perform the test on all detected GPU.
    """
    return cpp_device_test_all_gpu()
