# Copyright 2023 quocdang1998

cdef class Interpolator:
    """
    Interpolation on a multi-dimensional data.
    
    Wrapper of the class :cpp:class:`merlin::splint::Interpolator`
    """

    cdef CppInterpolator * core

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, grid=grid, values=values, methods=methods, n_threads=1)
        __init__(self, grid=grid, values=values, methods=methods, stream=stream, n_threads=1)
        Initializer.

        Parameters
        ----------
        grid: merlin.splint.CartesianGrid
            Cartesian grid to interpolate.
        methods: Sequence[merlin.splint.Method]
            Interpolation method to use for each dimension.
        """

        cdef uintptr_t dummy_ptr
        cdef Method dummy_method
        cdef CppArray * cpp_cpuvalues_array_ptr
        cdef CppParcel * cpp_cpuvalues_parcel_ptr
        cdef CppCartesianGrid * cpp_cartesiangrid_ptr
        cdef CppVector[Method] cpp_methods
        cdef uint64_t n_threads
        cdef object py_stream = Stream()
        cdef CppStream * cpp_stream_ptr

        if not kwargs:
            # default constructor called when no argument provided
            self.core = new CppInterpolator()
        elif kwargs.get("grid") is not None and kwargs.get("values") is not None and kwargs.get("methods") is not None:
            # get grid
            py_grid = kwargs.pop("grid")
            if not isinstance(py_grid, CartesianGrid):
                raise TypeError("Expected argument \"grid\" has type \"merlin.splint.CartesianGrid\".")
            dummy_ptr = py_grid.pointer()
            cpp_cartesiangrid_ptr = <CppCartesianGrid *>(dummy_ptr)
            # get methods
            py_methods = kwargs.pop("methods")
            cpp_methods = CppVector[Method](len(py_methods), Method.Linear)
            for i_m, m in enumerate(py_methods):
                dummy_method = m
                cpp_methods[i_m] = dummy_method
            # get values and dependance constructor
            py_values = kwargs.pop("values")
            # constructor on CPU
            if isinstance(py_values, Array):
                n_threads = 1
                if kwargs.get("n_threads") is not None:
                    n_threads = kwargs.pop("n_threads")
                dummy_ptr = py_values.pointer()
                cpp_cpuvalues_array_ptr = <CppArray *>(dummy_ptr)
                self.core = new CppInterpolator(dereference(cpp_cartesiangrid_ptr),
                                                dereference(cpp_cpuvalues_array_ptr),
                                                cpp_methods, n_threads)
            elif isinstance(py_values, Parcel):
                n_threads = 32
                if kwargs.get("n_threads") is not None:
                    n_threads = kwargs.pop("n_threads")
                if kwargs.get("stream") is not None:
                    py_stream = kwargs.pop("stream")
                    if not isinstance(py_stream, Stream):
                        raise TypeError("Expected argument \"stream\" has type \"merlin.cuda.Stream\".")
                dummy_ptr = py_stream.pointer()
                cpp_stream_ptr = <CppStream *>(dummy_ptr)
                dummy_ptr = py_values.pointer()
                cpp_cpuvalues_parcel_ptr = <CppParcel *>(dummy_ptr)
                self.core = new CppInterpolator(dereference(cpp_cartesiangrid_ptr),
                                                dereference(cpp_cpuvalues_parcel_ptr),
                                                cpp_methods, dereference(cpp_stream_ptr), n_threads)

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
        self.core = <CppInterpolator *>(ptr)

    cdef c_assign(self, CppInterpolator * new_core):
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

    def __dealloc__(self):
        del self.core
