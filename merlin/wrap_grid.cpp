// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include "merlin/array/array.hpp"          // merlin::array::Array
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid

namespace merlin {

// Wrap merlin::grid::CartesianGrid class
static void wrap_cartgrid(py::module & grid_module) {
    auto cartgrid_pyclass = py::class_<grid::CartesianGrid>(
        grid_module,
        "CartesianGrid",
        R"(
        Multi-dimensional Cartesian grid.

        Wrapper of :cpp:class:`merlin::grid::CartesianGrid`.
        )"
    );
    // constructor
    cartgrid_pyclass.def(
        py::init(
            [](py::list & grid_vectors) {
                Vector<DoubleVec> cpp_grid_vector(grid_vectors.size());
                std::uint64_t i_dim = 0;
                for (auto it = grid_vectors.begin(); it != grid_vectors.end(); ++it) {
                    cpp_grid_vector[i_dim++] = pyseq_to_vector<double>(it->cast<py::sequence>());
                }
                return new grid::CartesianGrid(cpp_grid_vector);
            }
        ),
        "Constructor from list of nodes on each dimension.",
        py::arg("grid_vectors")
    );
    // attributes
    cartgrid_pyclass.def_property_readonly(
        "ndim",
        [](const grid::CartesianGrid & self) { return self.ndim(); },
        "Get number of dimension."
    );
    cartgrid_pyclass.def_property_readonly(
        "shape",
        [](const grid::CartesianGrid & self) { return py::cast(self.shape()); },
        "Get shape."
    );
    cartgrid_pyclass.def_property_readonly(
        "size",
        [](const grid::CartesianGrid & self) { return self.size(); },
        "Get number of points in the grid."
    );
    cartgrid_pyclass.def_property_readonly(
        "num_nodes",
        [](const grid::CartesianGrid & self) { return self.num_nodes(); },
        "Get total number of nodes on all dimension."
    );
    cartgrid_pyclass.def(
        "get_grid_vector",
        [](const grid::CartesianGrid & self, std::uint64_t i_dim) { return vector_to_pylist(self.grid_vector(i_dim)); },
        "Get grid vector of a given dimension.",
        py::arg("i_dim")
    );
    // slicing operator
    cartgrid_pyclass.def(
        "get",
        [](const grid::CartesianGrid & self, std::uint64_t index) { return vector_to_pylist(self[index]); },
        "Get element at a given flatten index.",
        py::arg("index")
    );
    // get points
    cartgrid_pyclass.def(
        "get_points",
        [](const grid::CartesianGrid & self) { return new array::Array(self.get_points()); },
        "Get all points in the grid as a 2D array."
    );
    // representation
    cartgrid_pyclass.def(
        "__repr__",
        [](const grid::CartesianGrid & self) { return self.str(); }
    );
}

void wrap_grid(py::module & merlin_package) {
    // add grid submodule
    py::module grid_module = merlin_package.def_submodule("grid", "Grid for interpolation and regression.");
    // add classes
    wrap_cartgrid(grid_module);
}

}  // namespace merlin
