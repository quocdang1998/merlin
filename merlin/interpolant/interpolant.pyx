# Copyright 2023 quocdang1998

cdef class PolynomialInterpolant:
    """Polynomial interpolation."""

    cdef CppPolynomialInterpolant * core

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, grid=grid, values=values, method=merlin.interpolant.Method.Lagrange)
        Initializer.

        Parameters
        ----------
        grid: merlin.interpolant.CartesianGrid
            Cartesian grid to interpolate.
        values: merlin.array.Array
            Function values at grid points.
        method: merlin.interpolant.Method
            Method to interpolate.
        """

        cdef uintptr_t temp_ptr
        cdef CppCartesianGrid * p_grid
        cdef CppArray * p_values
        cdef Method method = Method.Lagrange

        if not kwargs:
            self.core = new CppPolynomialInterpolant()
        elif (kwargs.get("grid") is not None) and (kwargs.get("values") is not None):
            grid_arg = kwargs.pop("grid")
            if not isinstance(grid_arg, CartesianGrid):
                raise TypeError("Invalid argument grid: wrong type.")
            temp_ptr = grid_arg.pointer()
            p_grid = <CppCartesianGrid *>(temp_ptr)
            values_arg = kwargs.pop("values")
            if not isinstance(values_arg, Array):
                raise TypeError("Invalid argument values: wrong type.")
            temp_ptr = values_arg.pointer()
            p_values = <CppArray *>(temp_ptr)
            if kwargs.get("method") is not None:
                method = kwargs.pop("method")
            self.core = new CppPolynomialInterpolant(dereference(p_grid), dereference(p_values), method)

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
        self.core = <CppPolynomialInterpolant *>(ptr)

    cdef c_assign(self, CppPolynomialInterpolant * new_core):
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

    def __call__(self, point):
        cdef CppFloatvec c_point = CppFloatvec(len(point), <double>(0.0))
        for dim, coordinate in enumerate(point):
            c_point[dim] = <double>(coordinate)
        return self.core[0](c_point)

    def __dealloc__(self):
        del self.core
