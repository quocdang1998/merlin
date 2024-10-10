Polynomial serialization
========================

A multivariate polynomial consisting of monomials for regression is serialized as follows:

-  The first 8 bytes stores the number of dimension :math:`d`.

-  The next :math:`8d` bytes stores the max monomial order per axis. The total number of coefficients :math:`C` can be
   calculated by multiplying elements in the order per axis vector.

-  The following :math:`8C` bytes stores the coefficients.
