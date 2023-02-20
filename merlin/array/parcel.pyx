# Copyright 2022 quocdang1998

cdef class Parcel(_NdData):
    """Multi-dimensional array on CPU."""

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, shape)
        Initializer.

        Parameters
        ----------
        shape: Tuple[int]
            Shape of the array.
        """

        cdef CppIntvec array_shape

        if not kwargs:
            self.core = new CppArray()
        elif kwargs.get("shape") is not None:
            shape_arg = kwargs.pop("shape")
            if not isinstance(shape_arg, tuple):
                raise TypeError("Expected \"shape\" argument has type tuple.")
            array_shape = intvec_from_tuple(shape_arg)
            self.core = new CppArray(array_shape)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    def device(self):
        """Get the GPU holding the data."""
        cdef object result = Device()
        cdef CppParcel * dynamic_core = <CppParcel *>(self.core)
        cdef const CppDevice * device_ptr = &(dynamic_core.device())
        result.assign(<uintptr_t>(device_ptr))
        return result

    def transfer_data_to_gpu(self, Array src, object stream = Stream()):
        if not isinstance(stream, Stream):
            raise TypeError("Expected argument \"stream\" has type \"merlin.cuda.Stream\".")
        cdef uintptr_t str_uintptr = stream.pointer()
        cdef CppStream * stream_ptr = <CppStream *>(str_uintptr)
        cdef CppArray * src_ptr = <CppArray *>(src.core)
        cdef CppParcel * dynamic_core = <CppParcel *>(self.core)
        dynamic_core.transfer_data_to_gpu(dereference(src_ptr), dereference(stream_ptr))
