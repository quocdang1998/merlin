// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include <map>  // std::map

#include "merlin/array/array.hpp"          // merlin::array::Array
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/splint/interpolator.hpp"  // merlin::splint::Interpolator
#include "merlin/splint/tools.hpp"         // merlin::splint::Method

namespace merlin {

// wrap ProcessorType
static const std::map<std::string, ProcessorType> proctype_map = {
    {"cpu", ProcessorType::Cpu},
    {"gpu", ProcessorType::Gpu}
};

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
               const std::string & processor) {
                Vector<splint::Method> cpp_method(pyseq_to_vector<splint::Method>(method));
                return new splint::Interpolator(grid, values, cpp_method, proctype_map.at(processor));
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
            Index eval_shape;
            eval_shape.fill(0);
            eval_shape[0] = points.shape()[0];
            array::Array * p_eval_values = new array::Array(eval_shape);
            DoubleVec eval_values;
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
