# Copyright 2023 quocdang1998

cdef class CartesianGrid:
    """
    Multi-dimensional Cartesian grid.
    
    Wrapper of the class :cpp:class:`merlin::splint::CartesianGrid`
    """

    cdef CppCartesianGrid * core

    def __init__(self, **kwargs):
        """__init__(self)
        __init__(self, grid_vectors=grid_vectors)
        Initializer.

        Parameters
        ----------
        grid_vectors: Sequence[Sequence[float]]
            List of lists of grid nodes on each dimension.
        """

        cdef CppFloatvec cpp_gridvectors
        cdef CppIntvec cpp_gridshape
        cdef uint64_t num_nodes = 0
        cdef uint64_t i_node = 0

        if not kwargs:
            # default constructor called when no argument provided
            self.core = new CppCartesianGrid()
        elif kwargs.get("grid_vectors") is not None:
            # construct a Cartesian grid given its grid nodes on each dimension
            grid_vectors_arg = kwargs.pop("grid_vectors")
            # get grid shape
            cpp_gridshape = CppIntvec(len(grid_vectors_arg), 0)
            for i_vector, grid_vector in enumerate(grid_vectors_arg):
                cpp_gridshape[i_vector] = len(grid_vector)
            # get total number of nodes on each dimension
            for i_dim in range(cpp_gridshape.size()):
                num_nodes += cpp_gridshape[i_dim]
            # get grid nodes
            cpp_gridvectors = CppFloatvec(num_nodes, 0.0)
            for i_vector, grid_vector in enumerate(grid_vectors_arg):
                for i_element, element in enumerate(grid_vector):
                    cpp_gridvectors[i_node] = element
                    i_node += 1
            # initialization
            self.core = new CppCartesianGrid(move(cpp_gridvectors), move(cpp_gridshape))

        if kwargs:
            raise ValueError("Invalid keywords: " + ", ".join(k for k in kwargs.keys()))

    @property
    def ndim(self):
        """ndim(self)
        Get dimensions of the grid.
        """
        return self.core.ndim()

    @property
    def shape(self):
        """shape(self)
        Get the shape of the grid.
        """
        return list_from_intvec(self.core.shape())

    @property
    def size(self):
        """size(self)
        Get total number of points in the grid.
        """
        return self.core.size()

    @property
    def num_nodes(self):
        """num_nodes(self)
        Get total number of nodes on all dimension.
        """
        return self.core.num_nodes()

    def get_pt_index(self, object index):
        """get_pt_index(self, object index)
        Get point at a given index

        index: Sequence[int]
            Index vector of the desired point.
        """
        cdef CppIntvec cpp_index = intvec_from_iteratable(index)
        cdef CppFloatvec cpp_point = self.core[0][cpp_index]
        cdef list py_point = []
        for i in range(cpp_point.size()):
            py_point.append(cpp_point[i])
        return py_point

    def get_pt_cindex(self, uint64_t index):
        """get_pt_cindex(self, uint64_t index)
        Get point at a given index

        index: int
            Flatten index of the desired point.
        """
        cdef CppFloatvec cpp_point = self.core[0][index]
        cdef list py_point = []
        for i in range(cpp_point.size()):
            py_point.append(cpp_point[i])
        return py_point

    def __repr__(self):
        return PyUnicode_FromString(self.core.str().c_str())

    def assign(self, uintptr_t ptr):
        """assign(self, ptr)
        Assign pointer to a C++ object to the Python class wrapper.

        Parameters
        ----------
        ptr: int
            Pointer to C++ object to assign to current object in form of an unsigned integer.
        """
        del self.core
        self.core = <CppCartesianGrid *>(ptr)

    cdef c_assign(self, CppCartesianGrid * new_core):
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
