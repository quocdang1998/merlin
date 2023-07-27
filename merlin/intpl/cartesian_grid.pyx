# Copyright 2023 quocdang1998

cdef class CartesianGrid:
    """Cartesian grid."""

    cdef CppCartesianGrid * core

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, grid_vectors=grid_vectors)
        Initializer.

        Parameters
        ----------
        grid_vectors: Iteratable[Iterable[float]]
            List of lists of grid values.
        """

        cdef CppVector[CppVector[double]] cpp_gridvectors

        if not kwargs:
            self.core = new CppCartesianGrid()
        elif kwargs.get("grid_vectors") is not None:
            grid_vectors_arg = kwargs.pop("grid_vectors")
            cpp_gridvectors = CppVector[CppVector[double]](len(grid_vectors_arg), CppVector[double]())
            for i_vector, grid_vector in enumerate(grid_vectors_arg):
                cpp_gridvectors[i_vector] = CppVector[double](len(grid_vector), 0.0)
                for i_element, element in enumerate(grid_vector):
                    cpp_gridvectors[i_vector][i_element] = element
            self.core = new CppCartesianGrid(move(cpp_gridvectors))

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    def __repr__(self):
        return PyUnicode_FromString(self.core.str().c_str())

    def assign(self, uintptr_t ptr):
        """assign(self, ptr)
        Assign pointer to a C++ object to the Python class wrapper.

        Parameters
        ----------
        ptr: int
            Pointer to C++ object to assign to current object in form of an unsigned integer.
        """
        del self.core
        self.core = <CppCartesianGrid *>(ptr)

    cdef c_assign(self, CppCartesianGrid * new_core):
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
