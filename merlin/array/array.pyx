# Copyright 2022 quocdang1998

cdef class Array(NdData):
    """Array(merlin.array.NdData)
    Multi-dimensional array on CPU.

    Inherits from :class:`merlin.array.NdData`.
    """

    cdef object reference_array

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, array=array, copy=False)
        __init__(self, shape=shape)
        Initializer.

        Parameters
        ----------
        array: numpy.ndarray
            Numpy array of ``dtype=np.float64``.
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
            self.core = new CppArray(array_data, array_shape, array_strides, array_copy)
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

    def clone_data_from_gpu(self, Parcel src, object stream = Stream()):
        """clone_data_from_gpu(self, src, stream = Stream())
        Copy data from GPU array to CPU array.
        """
        if not isinstance(stream, Stream):
            raise TypeError("Expected argument \"stream\" has type \"merlin.cuda.Stream\".")
        cdef uintptr_t str_uintptr = stream.pointer()
        cdef CppStream * stream_ptr = <CppStream *>(str_uintptr)
        cdef CppParcel * src_ptr = <CppParcel *>(src.core)
        cdef CppArray * dynamic_core = <CppArray *>(self.core)
        dynamic_core.clone_data_from_gpu(dereference(src_ptr), dereference(stream_ptr))

    def extract_data_from_file(self, Stock src):
        """extract_data_from_file(self, src)
        Read data from serialized file.
        """
        cdef CppStock * src_ptr = <CppStock *>(src.core)
        cdef CppArray * dynamic_core = <CppArray *>(self.core)
        dynamic_core.extract_data_from_file(dereference(src_ptr))

    def __dealloc__(self):
        if self.reference_array is not None:
            Py_DECREF(self.reference_array)
