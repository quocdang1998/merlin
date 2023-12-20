// Copyright 2023 quocdang1998
#include "merlin/splint/interpolator.hpp"
#include "merlin/splint/tools.hpp"

#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/synchronizer.hpp"

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
               ProcessorType processor) {
                Vector<splint::Method> method_cpp(method.size());
                std::uint64_t i = 0;
                for (auto it = method.begin(); it != method.end(); ++it) {
                    method[i] = (*it).cast<splint::Method>();
                    i++;
                }
                return new splint::Interpolator(grid, values, method_cpp, processor);
            }
        ),
        "Construct from an array of values.",
        py::arg("grid"), py::arg("values"), py::arg("method"), py::arg("processor") = ProcessorType::Cpu
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
            floatvec eval_values = self.evaluate(points, n_threads);
            intvec eval_shape = {eval_values.size()};
            intvec eval_strides = {sizeof(double)};
            array::Array result(eval_values.data(), eval_shape, eval_strides, false);
            eval_values.data() = nullptr;
            return result;
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
