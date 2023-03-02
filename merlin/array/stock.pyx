# Copyright 2022 quocdang1998

cdef class Stock(NdData):
    """Multi-dimensional array serialized to a file."""

    def __init__(self, uint64_t offset=0, bint thread_safe=True, **kwargs):
        """__init__(self)
        __init__(self, filename)
        __init__(self, filename, shape)
        Initializer.

        Parameters
        ----------
        filename: str
            Name of the exported file to read/write.
        shape: Tuple[int]
            Shape of the array.

        If the shape argument is not provided, array shape is read from file. Otherwise, the file is prepared (created
        and resized if needed) to be able to hold the data with the precised shape.
        """

        cdef CppIntvec shape
        cdef str pyfilename
        cdef string filename

        if not kwargs:
            self.core = new CppStock()
        elif kwargs.get("filename") is not None:
            pyfilename = kwargs.pop("filename")
            filename = pyfilename.encode('UTF-8')
            if kwargs.get("shape") is not None:
                shape_arg = kwargs.pop("shape")
                if not isinstance(shape_arg, tuple):
                    raise TypeError("Expected \"shape\" argument has type tuple.")
                shape = intvec_from_tuple(shape_arg)
                self.core = new CppStock(filename, shape, offset, thread_safe)
            else:
                self.core = new CppStock(filename, offset, thread_safe)

    def record_data_to_file(self, Array src):
        """record_data_to_file(self, src)
        Wrtie data to serialized file.
        """
        cdef CppArray * src_ptr = <CppArray *>(src.core)
        cdef CppStock * dynamic_core = <CppStock *>(self.core)
        dynamic_core.record_data_to_file(dereference(src_ptr))
