// Copyright 2023 quocdang1998
#include "merlin/grid/cartesian_grid.hpp"

#include <vector>  // std::vector

#include "merlin/array/array.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace merlin {

// Wrap merlin::grid::CartesianGrid class
void wrap_cartgrid(py::module & grid_module) {
    auto cartgrid_pyclass = py::class_<grid::CartesianGrid>(
        grid_module,
        "CartesianGrid",
        R"(
        Multi-dimensional Cartesian grid.

        Wrapper of :cpp:class:`merlin::grid::CartesianGrid`.
        )"
    );
    // constructors
    cartgrid_pyclass.def(
        py::init([]() { return new grid::CartesianGrid(); }),
        "Default constructor."
    );
    cartgrid_pyclass.def(
        py::init(
            [](py::list & grid_vectors) {
                std::vector<std::vector<double>> grid_vector_cpp;
                grid_vector_cpp.reserve(grid_vectors.size());
                for (auto it : grid_vectors) {
                    grid_vector_cpp.push_back((*it).cast<std::vector<double>>());
                }
                Vector<floatvec> grid_vector_merlin(grid_vector_cpp.size());
                for(std::uint64_t i = 0; i < grid_vector_merlin.size(); i++) {
                    grid_vector_merlin[i].assign(grid_vector_cpp[i].data(), grid_vector_cpp[i].size());
                }
                return new grid::CartesianGrid(grid_vector_merlin);
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
        [](const grid::CartesianGrid & self) {
            const intvec & shape = self.shape();
            std::vector shape_cpp(shape.cbegin(), shape.cend());
            py::list shape_python = py::cast(shape_cpp);
            return shape_python;
        },
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
        [](const grid::CartesianGrid & self, std::uint64_t i_dim) {
            const floatvec grid_vector = self.grid_vector(i_dim);
            std::vector<double> grid_vector_cpp(grid_vector.cbegin(), grid_vector.cend());
            py::list grid_vector_python = py::cast(grid_vector_cpp);
            return grid_vector_python;
        },
        "Get grid vector of a given dimension.",
        py::arg("i_dim")
    );
    // slicing operator
    cartgrid_pyclass.def(
        "get",
        [](const grid::CartesianGrid & self, std::uint64_t index) {
            floatvec point = self[index];
            std::vector<double> point_cpp(point.begin(), point.end());
            py::list point_python = py::cast(point_cpp);
            return point_python;
        },
        "Get element at a given flatten index.",
        py::arg("index")
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
