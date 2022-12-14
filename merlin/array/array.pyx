# Copyright 2022 quocdang1998

cdef class Array(_NdData):
    """Multi-dimensional array on CPU."""

    def __init__(self, **kwargs):
        """__init__(self, **kwargs)
        Initializer.

        Numpy constructor
        -----------------
        array: numpy.ndarray
            Multi dimensional array of type ``np.float64``

        Shape constructor
        -----------------
        shape: Tuple[int]
            Shape of the array.
        """
        cdef CppIntvec shape

        cdef np.ndarray array_arg
        cdef int array_type

        if not kwargs:
            self.core = new CppArray()
        elif kwargs.get("array") is not None:
            array_arg = kwargs.pop("array")
            array_type = np.PyArray_TYPE(array_arg)
            if array_type != np.NPY_DOUBLE:
                raise TypeError("Expected Numpy array of type \"double\"")

        elif kwargs.get("shape") is not None:
            arg = kwargs.pop("shape")
            if not isinstance(arg, tuple):
                raise TypeError("Expected \"shape\" argument has type tuple.")
            shape = intvec_from_tuple(arg)
            self.core = new CppArray(shape)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))



    def __dealloc__(self):
        del self.core
