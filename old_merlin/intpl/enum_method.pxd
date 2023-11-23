# Copyright 2023 quocdang1998

cdef extern from "merlin/intpl/interpolant.hpp":

    cpdef enum class Method "merlin::intpl::Method":
        """Method for polynomial interpolation.

        *Values*

         - ``Lagrange``: Lagrange method.
         - ``Newton``: Newton method.
        """
        Lagrange,
        Newton
