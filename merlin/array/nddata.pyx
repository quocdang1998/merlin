# Copyright 2022 quocdang1998

cdef class _NdData:
    """Abstract class of N-dim array."""

    cdef CppNdData * core

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Shape constructor
        -----------------
        shape: Tuple[int]
            Shape of the array.
        """
        cdef CppIntvec shape

        if not kwargs:
            self.core = new CppNdData()
        elif kwargs.get("shape") is not None:
            arg = kwargs.pop("shape")
            if not isinstance(arg, tuple):
                raise TypeError("Expected \"shape\" argument has type tuple.")
            shape = intvec_from_tuple(arg)
            self.core = new CppNdData(shape)

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
        self.core = <CppNdData *>(ptr)

    cdef c_assign(self, CppNdData * new_core):
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

    def data(self):
        """Get pointer to data."""
        return <uintptr_t>(self.core.data())

    def ndim(self):
        """Get number of dimension."""
        return self.core.ndim()

    def shape(self):
        """Get shape of array."""
        return tuple_from_intvec(self.core.shape())

    def strides(self):
        """Get strides of array."""
        return tuple_from_intvec(self.core.strides())

    def size(self):
        """Get number of elements in array."""
        return self.core.size()

    def get(self, object index):
        cdef CppIntvec idx_vector
        cdef uint64_t idx_int
        if isinstance(index, tuple):
            idx_vector = intvec_from_tuple(index)
            return self.core.get(idx_vector)
        elif isinstance(index, int):
            idx_int = index
            return self.core.get(idx_int)
        else:
            raise TypeError("Expected an integer or a tuple.")

    def set(self, object index, double value):
        cdef CppIntvec idx_vector
        cdef uint64_t idx_int
        if isinstance(index, tuple):
            idx_vector = intvec_from_tuple(index)
            return self.core.set(idx_vector, value)
        elif isinstance(index, int):
            idx_int = index
            return self.core.set(idx_int, value)
        else:
            raise TypeError("Expected an integer or a tuple.")

    def __dealloc__(self):
        del self.core
