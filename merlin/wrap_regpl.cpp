// Copyright 2024 quocdang1998
#include "merlin/regpl/polynomial.hpp"
#include "merlin/regpl/regressor.hpp"

#include <vector>

#include "merlin/array/array.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "py_common.hpp"

namespace merlin {

// Wrap merlin::regpl::Polynomial class
void wrap_polynomial(py::module & regpl_module) {
    auto polynomial_pyclass = py::class_<regpl::Polynomial>(
        regpl_module,
        "Polynomial",
        R"(
        Multi-variate polynomial.

        Wrapper of :cpp:class:`merlin::regpl::Polynomial`.
        )"
    );
    // constructors
    polynomial_pyclass.def(
        py::init([]() { return new regpl::Polynomial(); }),
        "Default constructor."
    );
    polynomial_pyclass.def(
        py::init(
            [](py::list & order_per_dim) {
                std::vector<std::uint64_t> order_per_dim_cpp = order_per_dim.cast<std::vector<std::uint64_t>>();
                return new regpl::Polynomial(intvec(order_per_dim_cpp.data(), order_per_dim_cpp.size()));
            }
        ),
        "Constructor of an empty polynomial from order per dimension.",
        py::arg("order_per_dim")
    );
    polynomial_pyclass.def(
        py::init(
            [](const array::Array & coeffs) {
                regpl::Polynomial * result = new regpl::Polynomial(coeffs.shape());
                for (std::uint64_t i = 0; i < coeffs.size(); i++) {
                    result->coeff()[i] = coeffs.get(i);
                }
                return result;
            }
        ),
        "Constructor from coefficients array.",
        py::arg("coeffs")
    );
    polynomial_pyclass.def(
        py::init(
            [](py::list & coeff_array, py::list & order_per_dim, py::list & term_index) {
                floatvec cpp_coeff = pylist_to_fvec(coeff_array);
                intvec cpp_order = pylist_to_ivec(order_per_dim);
                intvec cpp_term = pylist_to_ivec(term_index);
                return new regpl::Polynomial(cpp_coeff, cpp_order, cpp_term);
            }
        ),
        "Constructor of a sparse polynomial from coefficients.",
        py::arg("coeff_array"), py::arg("order_per_dim"), py::arg("term_index")
    );
    // attributes
    polynomial_pyclass.def(
        "order",
        [](const regpl::Polynomial & self) {
            const intvec & order = self.order();
            std::vector<std::uint64_t> order_cpp(order.cbegin(), order.cend());
            py::list order_python = py::cast(order_cpp);
            return order_python;
        },
        "Get order per dimension of the polynomial."
    );
    // get Vandermonde matrix
    polynomial_pyclass.def(
        "calc_vandermonde",
        [](const regpl::Polynomial & self, const array::Array & grid_points, std::uint64_t n_threads) {
            return new array::Array(self.calc_vandermonde(grid_points, n_threads));
        },
        "Calculate Vandermonde matrix.",
        py::arg("grid_points"), py::arg("n_threads") = 1
    );
    // serialization
    polynomial_pyclass.def(
        "serialize",
        [](const regpl::Polynomial & self, const std::string & fname) {
            self.serialize(fname);
        },
        "Save polynomial into a file.",
        py::arg("fname")
    );
    polynomial_pyclass.def(
        "deserialize",
        [](regpl::Polynomial & self, const std::string & fname) {
            self.deserialize(fname);
        },
        "Read polynomial from a file.",
        py::arg("fname")
    );
    // representation
    polynomial_pyclass.def(
        "__repr__",
        [](const regpl::Polynomial & self) { return self.str(); }
    );
}

// Wrap merlin::regpl::Regressor class
void wrap_regressor(py::module & regpl_module) {
    auto regressor_pyclass = py::class_<regpl::Regressor>(
        regpl_module,
        "Regressor",
        R"(
        Launch polynomial regression.

        Wrapper of :cpp:class:`merlin::regpl::Regressor`.
        )"
    );
    // constructors
    regressor_pyclass.def(
        py::init([]() { return new regpl::Regressor(); }),
        "Default constructor."
    );
    regressor_pyclass.def(
        py::init(
            [](const regpl::Polynomial & polynom, const std::string & proc_type) {
                return new regpl::Regressor(polynom, proctype_map.at(proc_type));
            }
        ),
        "Constructor from polynomial object.",
        py::arg("polynom"), py::arg("proc_type") = "cpu"
    );
    // evaluate
    regressor_pyclass.def(
        "evaluate",
        [](regpl::Regressor & self, const array::Array & points, std::uint64_t n_threads) {
            intvec eval_values_shape = {points.shape()[0]};
            array::Array * eval_values = new array::Array(eval_values_shape);
            self.evaluate(points, eval_values->data(), n_threads);
            return eval_values;
        },
        "Evaluate regression by CPU parallelism.",
        py::arg("points"), py::arg("n_threads") = 1
    );
    // synchronize
    regressor_pyclass.def(
        "synchronize",
        [](regpl::Regressor & self) { self.synchronize(); },
        "Force the current CPU to wait until all asynchronous tasks have finished."
    );
}

void wrap_regpl(py::module & merlin_package) {
    // add regpl submodule
    py::module regpl_module = merlin_package.def_submodule("regpl", "Multi-dimensional polynomial regression API.");
    // add classes
    wrap_polynomial(regpl_module);
    wrap_regressor(regpl_module);
}

}  // namespace merlin
