// Copyright 2024 quocdang1998
#include "py_api.hpp"

#include <optional>  // std::optional

#include "merlin/array/array.hpp"          // merlin::array::Array
#include "merlin/array/parcel.hpp"         // merlin::array::Parcel
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/regpl/polynomial.hpp"     // merlin::regpl::Polynomial
#include "merlin/regpl/regressor.hpp"      // merlin::regpl::Regressor
#include "merlin/regpl/vandermonde.hpp"    // merlin::regpl::Vandermonde

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
    // constructor
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
    // attributes
    polynomial_pyclass.def(
        "order",
        [](const regpl::Polynomial & self) { return array_to_pylist(self.order(), self.ndim()); },
        "Get order per dimension of the polynomial."
    );
    // serialization
    polynomial_pyclass.def(
        "save",
        [](const regpl::Polynomial & self, const std::string & fname) {
            self.save(fname);
        },
        "Save polynomial into a file.",
        py::arg("fname")
    );
    polynomial_pyclass.def(
        "load",
        [](regpl::Polynomial & self, const std::string & fname) {
            self.load(fname);
        },
        "Read polynomial from a file.",
        py::arg("fname")
    );
    // representation
    polynomial_pyclass.def(
        "__repr__",
        [](const regpl::Polynomial & self) { return self.str(); }
    );
    // empty polynomial
    regpl_module.def(
        "new_polynom",
        [](py::sequence & order, std::optional<py::sequence> & coeff, std::optional<py::sequence> & term_idx) {
            regpl::Polynomial * poly = new regpl::Polynomial(pyseq_to_array<std::uint64_t>(order));
            if (coeff.has_value() && term_idx.has_value()) {
                poly->set(pyseq_to_vector<double>(coeff.value()).data(),
                          pyseq_to_vector<std::uint64_t>(term_idx.value()));
            }
            return poly;
        },
        R"(
        Create a sparse polynomial.

        A sparse polynomial is a polynomial of which most of the coefficients are zeros.

        Parameters
        ----------
        order : Sequence[int]
            Max polynomial order per dimension.
        coeff : Sequence[float], default=None
            Coefficient data of each term index.
        term_idx : Sequence[int], default=None
            Index of terms to assign (in C-contiguous order).)",
        py::arg("order"), py::arg("coeff") = py::none(), py::arg("term_idx") = py::none()
    );
}

// Wrap merlin::regpl::Vandermonde class
void wrap_vandermonde(py::module & regpl_module) {
    auto vandermonde_pyclass = py::class_<regpl::Vandermonde>(
        regpl_module,
        "Vandermonde",
        R"(
        Vandermonde matrix of a polynomial and a grid.

        Wrapper of :cpp:class:`merlin::regpl::Vandermonde`.
        )"
    );
    // solve for polynomial
    vandermonde_pyclass.def(
        "solve",
        [] (regpl::Vandermonde & self, array::Array & flatten_data) {
            if (flatten_data.ndim() != 1) {
                Fatal<std::invalid_argument>("Input data must be a vector.\n");
            }
            if (!(flatten_data.is_c_contiguous())) {
                Fatal<std::invalid_argument>("Input vector must be contiguous.\n");
            }
            if (flatten_data.shape()[0] != self.num_points()) {
                Fatal<std::invalid_argument>("Inconsistent size between the input grid and the data provided.\n");
            }
            regpl::Polynomial * result = new regpl::Polynomial();
            self.solve(flatten_data.data(), *result);
            return result;
        },
        py::arg("flatten_data")
    );
    // constructors
    regpl_module.def(
        "create_vandermonde",
        [] (py::sequence & order, const grid::CartesianGrid & grid, std::uint64_t n_threads) {
            return new regpl::Vandermonde(pyseq_to_array<std::uint64_t>(order), grid, n_threads);
        },
        R"(
        Constructor from a full polynomial and Cartesian grid.

        Parameters
        ----------
        order : Sequence[int]
            Max polynomial order per dimension.
        grid : merlin.grid.CartesianGrid
            Cartesian grid of points.
        n_threads : int, default=1
            Number of threads to perform the calculation.)",
        py::arg("order"), py::arg("grid"), py::arg("n_threads") = 1
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
        py::init(
            [](regpl::Polynomial & polynom, Synchronizer & synchronizer) {
                return new regpl::Regressor(std::move(polynom), synchronizer);
            }
        ),
        "Constructor from polynomial object.",
        py::arg("polynom"), py::arg("synchronizer")
    );
    // evaluate
    regressor_pyclass.def(
        "evaluate_cpu",
        [](regpl::Regressor & self, const array::Array & points, std::uint64_t n_threads) {
            DoubleVec result(points.shape()[0]);
            self.evaluate(points, result, n_threads);
            double * data = std::exchange(result.data(), nullptr);
            return make_wrapper_array<double>(data, result.size());
        },
        "Evaluate regression by CPU parallelism.",
        py::arg("points"), py::arg("n_threads") = 1
    );
    regressor_pyclass.def(
        "evaluate_gpu",
        [](regpl::Regressor & self, const array::Parcel & points, std::uint64_t n_threads) {
            DoubleVec result(points.shape()[0]);
            self.evaluate(points, result, n_threads);
            double * data = std::exchange(result.data(), nullptr);
            return make_wrapper_array<double>(data, result.size());
        },
        "Evaluate regression by GPU parallelism.",
        py::arg("points"), py::arg("n_threads") = 32
    );
}

void wrap_regpl(py::module & merlin_package) {
    // add regpl submodule
    py::module regpl_module = merlin_package.def_submodule("regpl", "Multi-dimensional polynomial regression API.");
    // add classes
    wrap_polynomial(regpl_module);
    wrap_regressor(regpl_module);
    wrap_vandermonde(regpl_module);
}

}  // namespace merlin
