# Copyright 2022 quocdang1998

cdef class Array(_NdData):
    """Multi-dimensional array on CPU."""

    cdef object reference_array

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, array, copy)
        __init__(self, shape)
        Initializer.

        Parameters
        ----------
        array: numpy.ndarray
            Multi dimensional array of type ``np.float64``.
        copy: bool, optional
            Copy data from old array to new array. Default to ``False``.
        shape: Tuple[int]
            Shape of the array.
        """

        cdef np.ndarray array_arg
        cdef int array_type
        cdef double * array_data
        cdef uint64_t array_ndim
        cdef CppIntvec array_shape
        cdef CppIntvec array_strides
        cdef bint array_copy = False

        self.reference_array = None

        if not kwargs:
            self.core = new CppArray()
        elif kwargs.get("array") is not None:
            array_arg = kwargs.pop("array")
            array_type = np.PyArray_TYPE(array_arg)
            if array_type != np.NPY_DOUBLE:
                raise TypeError("Expected Numpy array of type \"np.double\"")
            array_data = <double *>(np.PyArray_DATA(array_arg))
            array_ndim = np.PyArray_NDIM(array_arg)
            array_shape = CppIntvec(np.PyArray_DIMS(array_arg), array_ndim)
            array_strides = CppIntvec(np.PyArray_STRIDES(array_arg), array_ndim)
            if kwargs.get("copy") is not None:
                array_copy = kwargs.pop("copy")
            self.core = new CppArray(array_data, array_ndim, array_shape.data(), array_strides.data(), array_copy)
            if not array_copy:
                self.reference_array = array_arg
                Py_INCREF(self.reference_array)
        elif kwargs.get("shape") is not None:
            shape_arg = kwargs.pop("shape")
            if not isinstance(shape_arg, tuple):
                raise TypeError("Expected \"shape\" argument has type tuple.")
            array_shape = intvec_from_tuple(shape_arg)
            self.core = new CppArray(array_shape)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    @property
    def __array_interface__(self):
        return {
            "data": (self.data, False),
            "shape": self.shape,
            "strides": self.strides,
            "typestr": "f8",
            "version": 3
        }

    def __dealloc__(self):
        if self.reference_array is not None:
            Py_DECREF(self.reference_array)