Array serialization
===================

The structure of multi-dimensional array in Merlin inherits the one of NumPy. An array is represented by:

-  A pointer refers to the first element of the array.

-  An integer represents the number of dimensions.

-  A shape vector stores the size of each dimension. It is required for manipulating multi-dimensional index.

-  A stride vector contains the number of memory locations (or bytes) between the beginnings of successive array
   elements along a specific dimension. It allows to determine the address of a specific element in the array.

-  A boolean indicates whether memory de-allocation should be called within the destructor or not.

In Merlin, the default convention is C-contiguous array, i.e. elements of the array is placed side-by-side without any
gaps in between, and the stride of the last dimension is equal to the size of the element. This is selected as the data
arrangement of the serialized file for reading/writing data.

In a C-contiguous stock file (see :cpp:class:`merlin::array::Stock`), the binary layout is as follow:

-  The first 8 bytes stores the number of dimension :math:`d`. Since the max number of dimension is fixed by the
   compile-time constant :cpp:member:`merlin::max_dim`, this number can serve as the endianness indicator.

-  The next :math:`8*d` bytes stores the shape vector the array.

-  The following bytes stores the data of the file arranged in C-contiguous order.

Developer of other projects can read and manipulate data in the stock file through Merlin's API. Using C/C++ IO library
such as ``std::fread`` / ``std::fwrite`` or ``std::istream::read`` / ``std::ostream::write`` is also possible.
