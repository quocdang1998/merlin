Polynomial serialization
========================

A multivariate polynomial consisting of monomials for regression is serialized as follows:

-  The first 8 bytes stores the number of dimension :math:`d`.

-  The next 8 bytes stores the total number of coefficients :math:`C`.

-  The next :math:`8d` bytes stores the max monomial order per axis.

-  The following :math:`8C` bytes stores the coefficients.
