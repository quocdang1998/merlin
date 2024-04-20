// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include "merlin/array/array.hpp"          // merlin::array::Array
#include "merlin/array/parcel.hpp"         // merlin::array::Parcel
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/splint/interpolator.hpp"  // merlin::splint::Interpolator
#include "merlin/splint/tools.hpp"         // merlin::splint::Method

namespace merlin {

// Wrap merlin::splint::Method enum
static void wrap_method(py::module & splint_module) {
    auto method_pyenum = py::enum_<splint::Method>(
        splint_module,
        "Method",
        "Wrapper of :cpp:enum:`merlin::splint::Method`"
    );
    method_pyenum.value("Linear", splint::Method::Linear);
    method_pyenum.value("Lagrange", splint::Method::Lagrange);
    method_pyenum.value("Newton", splint::Method::Newton);
}

// Wrap merlin::splint::Interpolator class
static void wrap_interpolator(py::module & splint_module) {
    auto interpolator_pyclass = py::class_<splint::Interpolator>(
        splint_module,
        "Interpolator",
        R"(
        Interpolation on a multi-dimensional data.

        Wrapper of :cpp:class:`merlin::splint::Interpolator`.
        )"
    );
    // constructor
    interpolator_pyclass.def(
        py::init(
            [](const grid::CartesianGrid & grid, const array::Array & values, py::list & method,
               Synchronizer & synchronizer) {
                Vector<splint::Method> cpp_method(pyseq_to_vector<splint::Method>(method));
                return new splint::Interpolator(grid, values, cpp_method.data(), synchronizer);
            }
        ),
        "Construct from an array of values.",
        py::arg("grid"), py::arg("values"), py::arg("method"), py::arg("synchronizer")
    );
    // attributes
    interpolator_pyclass.def_property_readonly(
        "gpu_id",
        [](const splint::Interpolator & self) { return self.gpu_id(); },
        "Get GPU ID on which the memory is allocated. Return ``-1`` if executed on CPU."
    );
    // build coefficients
    interpolator_pyclass.def(
        "build_coefficients",
        [](splint::Interpolator & self, std::uint64_t n_threads) { return self.build_coefficients(n_threads); },
        "Calculate interpolation coefficients based on provided method.",
        py::arg("n_threads") = 1
    );
    // evaluate interpolation
    interpolator_pyclass.def(
        "evaluate_cpu",
        [](splint::Interpolator & self, const array::Array & points, std::uint64_t n_threads) {
            DoubleVec eval_values(points.shape()[0]);
            self.evaluate(points, eval_values, n_threads);
            double * data = std::exchange(eval_values.data(), nullptr);
            return make_wrapper_array<double>(data, eval_values.size());
        },
        "Evaluate interpolation by CPU.",
        py::arg("points"), py::arg("n_threads") = 1
    );
    interpolator_pyclass.def(
        "evaluate_gpu",
        [](splint::Interpolator & self, const array::Parcel & points, std::uint64_t n_threads) {
            DoubleVec eval_values(points.shape()[0]);
            self.evaluate(points, eval_values, n_threads);
            double * data = std::exchange(eval_values.data(), nullptr);
            return make_wrapper_array<double>(data, eval_values.size());
        },
        "Evaluate interpolation by GPU.",
        py::arg("points"), py::arg("n_threads") = 32
    );
    // representation
    interpolator_pyclass.def(
        "__repr__",
        [](const splint::Interpolator & self) { return self.str(); }
    );
}

void wrap_splint(py::module & merlin_package) {
    // add array submodule
    py::module splint_module = merlin_package.def_submodule("splint", "Parallel polynomial interpolation library.");
    // add classes
    wrap_method(splint_module);
    wrap_interpolator(splint_module);
}

}  // namespace merlin
