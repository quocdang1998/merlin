// Copyright 2023 quocdang1998
#include "merlin/synchronizer.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace merlin {

// Wrap ProcessorType enum
void wrap_proctype(py::module & merlin_package) {
    auto processor_type_pyenum = py::enum_<ProcessorType>(
        merlin_package,
        "ProcessorType",
        "Wrapper of :cpp:enum:`merlin::ProcessorType`"
    );
    processor_type_pyenum.value("Cpu", ProcessorType::Cpu);
    processor_type_pyenum.value("Gpu", ProcessorType::Gpu);
}

}  // namespace merlin
