# Copyright 2023 quocdang1998

cdef extern from "merlin/interpolant/interpolant.hpp":

    cpdef enum class Method "merlin::interpolant::Method":
        """Method for polynomial interpolation.

        *Values*

         - ``Lagrange``: Lagrange method.
         - ``Newton``: Newton method.
        """
        Lagrange,
        Newton
