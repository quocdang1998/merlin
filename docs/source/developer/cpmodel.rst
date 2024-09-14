CP model
========

The serialization scheme of CP model is as follows:

-  The first 8 bytes stores the model rank :math:`r`.

-  The next 8 bytes stores the number of dimension :math:`d`.

-  The following 8 bytes stores the total number of parameters of the model :math:`C`.

-  The next :math:`8d` bytes stores the product of the number of elements per axis and the rank of the model.

-  The last :math:`8C` bytes stores the parameters.
