# Copyright 2023 quocdang1998

cdef class Interpolator:
    """
    Interpolation on a multi-dimensional data.
    
    Wrapper of the class :cpp:class:`merlin::splint::Interpolator`
    """

    cdef CppInterpolator * core

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, grid=grid, coeff=coeff, methods=methods, n_threads=1)
        Initializer.

        Parameters
        ----------
        grid: merlin.splint.CartesianGrid
            Cartesian grid to interpolate.
        """

        if not kwargs:
            # default constructor called when no argument provided
            self.core = new CppInterpolator()

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
