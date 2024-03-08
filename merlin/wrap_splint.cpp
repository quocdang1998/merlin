// Copyright 2023 quocdang1998
#include "merlin/splint/interpolator.hpp"
#include "merlin/splint/tools.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/synchronizer.hpp"

#include "py_common.hpp"

namespace py = pybind11;

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
void wrap_interpolator(py::module & splint_module) {
    auto interpolator_pyclass = py::class_<splint::Interpolator>(
        splint_module,
        "Interpolator",
        R"(
        Interpolation on a multi-dimensional data.

        Wrapper of :cpp:class:`merlin::splint::Interpolator`.
        )"
    );
    // constructors
    interpolator_pyclass.def(
        py::init([]() { return new splint::Interpolator(); }),
        "Default constructor."
    );
    interpolator_pyclass.def(
        py::init(
            [](const grid::CartesianGrid & grid, const array::Array & values, py::list & method,
               const std::string & processor) {
                Vector<splint::Method> method_cpp(method.size());
                std::uint64_t i = 0;
                for (auto it = method.begin(); it != method.end(); ++it) {
                    method_cpp[i] = (*it).cast<splint::Method>();
                    i++;
                }
                return new splint::Interpolator(grid, values, method_cpp, proctype_map[processor]);
            }
        ),
        "Construct from an array of values.",
        py::arg("grid"), py::arg("values"), py::arg("method"), py::arg("processor") = "cpu"
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
        "evaluate",
        [](splint::Interpolator & self, const array::Array & points, std::uint64_t n_threads) {
            intvec eval_shape = {points.shape()[0]};
            array::Array * p_eval_values = new array::Array(eval_shape);
            floatvec eval_values;
            eval_values.assign(p_eval_values->data(), p_eval_values->size());
            self.evaluate(points, eval_values, n_threads);
            return p_eval_values;
        },
        "Evaluate interpolation by CPU.",
        py::arg("points"), py::arg("n_threads") = 1
    );
    // synchronization
    interpolator_pyclass.def(
        "synchronize",
        [](splint::Interpolator & self) { return self.synchronize(); },
        "Force the current CPU to wait until all asynchronous tasks have finished."
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
