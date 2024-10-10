CP model
========

The serialization scheme of CP model is as follows:

-  The first 8 bytes stores the model rank :math:`r`.

-  The next 8 bytes stores the number of dimension :math:`d`.

-  The next :math:`8d` bytes stores the shape of the model, i.e. the number of elements per axis. The total number of
   parameters of the model :math:`C` is the product of elements of the shape vector.

-  The last :math:`8C` bytes stores the parameters.
