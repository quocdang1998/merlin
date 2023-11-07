# Copyright 2022 quocdang1998

cdef class Parcel(NdData):
    """Parcel(merlin.array.NdData)
    Multi-dimensional array on GPU.

    Inherits from :class:`merlin.array.NdData`.
    """

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
            self.core = new CppParcel()
        elif kwargs.get("shape") is not None:
            shape_arg = kwargs.pop("shape")
            array_shape = intvec_from_iteratable(shape_arg)
            self.core = new CppParcel(array_shape)

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    @property
    def device(self):
        """Get the GPU holding the data."""
        cdef object result = Device()
        cdef CppParcel * dynamic_core = <CppParcel *>(self.core)
        cdef const CppDevice * device_ptr = &(dynamic_core.device())
        result.assign(<uintptr_t>(device_ptr))
        return result

    def transfer_data_to_gpu(self, Array src, object stream = Stream()):
        """transfer_data_to_gpu(self, src, stream = merlin.cuda.Stream())
        Transfer data from CPU to GPU.
        """
        if not isinstance(stream, Stream):
            raise TypeError("Expected argument \"stream\" has type \"merlin.cuda.Stream\".")
        cdef uintptr_t str_uintptr = stream.pointer()
        cdef CppStream * stream_ptr = <CppStream *>(str_uintptr)
        cdef CppArray * src_ptr = <CppArray *>(src.core)
        cdef CppParcel * dynamic_core = <CppParcel *>(self.core)
        dynamic_core.transfer_data_to_gpu(dereference(src_ptr), dereference(stream_ptr))
