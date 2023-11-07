# Copyright 2023 quocdang1998

cdef extern from "merlin/splint/tools.hpp":

    cpdef enum class Method "merlin::splint::Method":
        """Interpolation method.

        See :cpp:enum:`merlin::splint::Method`

        *Values*

         - ``Linear``: Linear interpolation
         - ``Lagrange``: Polynomial interpoaltion by Lagrange method.
         - ``Newton``: Polynomial interpoaltion by Newton method.
        """
        Linear,
        Lagrange,
        Newton
