// Copyright 2023 quocdang1998
#include "merlin/cuda_interface.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/enum_wrapper.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/cuda/stream.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace merlin {

// Wrap enums
static void wrap_enums(py::module & cuda_module) {
    // wrap cuda::DeviceLimit
    auto device_limit_pyenum = py::enum_<cuda::DeviceLimit>(
        cuda_module,
        "DeviceLimit",
        "Wrapper of :cpp:class:`merlin::cuda::DeviceLimit`"
    );
    device_limit_pyenum.value("StackSize", cuda::DeviceLimit::StackSize);
    device_limit_pyenum.value("PrintfSize", cuda::DeviceLimit::PrintfSize);
    device_limit_pyenum.value("HeapSize", cuda::DeviceLimit::HeapSize);
    device_limit_pyenum.value("SyncDepth", cuda::DeviceLimit::SyncDepth);
    device_limit_pyenum.value("LaunchPendingCount", cuda::DeviceLimit::LaunchPendingCount);
    // wrap cuda::EventCategory
    auto event_category_pyenum = py::enum_<cuda::EventCategory>(
        cuda_module,
        "EventCategory",
        "Wrapper of :cpp:class:`merlin::cuda::EventCategory`"
    );
    event_category_pyenum.value("DefaultEvent", cuda::EventCategory::DefaultEvent);
    event_category_pyenum.value("BlockingSyncEvent", cuda::EventCategory::BlockingSyncEvent);
    event_category_pyenum.value("DisableTimingEvent", cuda::EventCategory::DisableTimingEvent);
    event_category_pyenum.value("InterprocessEvent", cuda::EventCategory::InterprocessEvent);
    // wrap cuda::EventWaitFlag
    auto event_wait_pyenum = py::enum_<cuda::EventWaitFlag>(
        cuda_module,
        "EventWaitFlag",
        "Wrapper of :cpp:class:`merlin::cuda::EventWaitFlag`"
    );
    event_wait_pyenum.value("DefaultEvent", cuda::EventWaitFlag::Default);
    event_wait_pyenum.value("BlockingSyncEvent", cuda::EventWaitFlag::External);
}

// Wrap merlin::cuda::Device
static void wrap_device(py::module & cuda_module) {
    auto device_pyclass = py::class_<cuda::Device>(
        cuda_module,
        "Device",
        R"(
        CUDA GPU management.

        Wrapper of :cpp:class:`merlin::cuda::Device`.)"
    );
    // constructors
    device_pyclass.def(
        py::init([](){ return new cuda::Device(); }),
        "Default constructor."
    );
    device_pyclass.def(
        py::init([](int id){ return new cuda::Device(id); }),
        "Constructor from GPU ID.",
        py::arg("id")
    );
    // members
    device_pyclass.def_property(
        "id",
        [](cuda::Device & device) { return device.id(); },
        [](cuda::Device & device, int new_id) { device.id() = new_id; }
    );
    // query
    device_pyclass.def(
        "print_specification",
        [](cuda::Device & device) { device.print_specification(); },
        "Print GPU specifications."
    );
    device_pyclass.def(
        "test_gpu",
        [](cuda::Device & device) { return device.test_gpu(); },
        "Test functionality of GPU."
    );
    device_pyclass.def_static(
        "get_current_gpu",
        []() { return cuda::Device::get_current_gpu(); },
        "Get current GPU."
    );
    device_pyclass.def_static(
        "get_num_gpu",
        []() { return cuda::Device::get_num_gpu(); },
        "Get total number of CUDA capable GPU."
    );
    // action
    device_pyclass.def(
        "set_as_current",
        [](cuda::Device & device) { device.set_as_current(); },
        "Set the GPU as current device."
    );
    device_pyclass.def_static(
        "get_limit",
        [](cuda::DeviceLimit limit) { return cuda::Device::limit(limit); },
        "Get setting limits of the current GPU.",
        py::arg("limit")
    );
    device_pyclass.def_static(
        "set_limit",
        [](cuda::DeviceLimit limit, std::uint64_t size) { return cuda::Device::limit(limit, size); },
        "Set setting limits of the current GPU.",
        py::arg("limit"), py::arg("size")
    );
    device_pyclass.def_static(
        "reset_all",
        []() { return cuda::Device::reset_all(); },
        "Destroy all allocations and reset the state of the current GPU."
    );
    // representation
    device_pyclass.def(
        "__repr__",
        [](cuda::Device & device) { return device.str(); }
    );
}

// Wrap merlin::cuda::Event
static void wrap_event(py::module & cuda_module) {
    auto event_pyclass = py::class_<cuda::Event>(
        cuda_module,
        "Event",
        R"(
        CUDA event.

        Wrapper of :cpp:class:`merlin::cuda::Event`.)"
    );
    // constructor
    event_pyclass.def(
        py::init([](unsigned int category) { return new cuda::Event(category); }),
        "Construct CUDA event from flag.",
        py::arg("category") = cuda::EventCategory::DefaultEvent
    );
    // attributes
    event_pyclass.def(
        "get_event_ptr",
        [](cuda::Event & event) { return event.get_event_ptr(); },
        "Get pointer to CUDA event."
    );
    event_pyclass.def(
        "get_category",
        [](cuda::Event & event) { return event.get_category(); },
        "Get setting flag of the event."
    );
    event_pyclass.def(
        "get_gpu",
        [](cuda::Event & event) { return event.get_gpu(); },
        "Get GPU associated to the event."
    );
    // query
    event_pyclass.def(
        "is_complete",
        [](cuda::Event & event) { return event.is_complete(); },
        "Query the status of all work currently captured by event."
    );
    event_pyclass.def(
        "check_cuda_context",
        [](cuda::Event & event) { event.check_cuda_context(); },
        "Check validity of GPU and context."
    );
    // operation
    event_pyclass.def(
        "synchronize",
        [](cuda::Event & event) { event.synchronize(); },
        "Block the CPU process until the event occurs."
    );
    event_pyclass.def(
        "__sub__",
        [](const cuda::Event & ev_1, const cuda::Event & ev_2) { return ev_1 - ev_2; },
        py::is_operator()
    );
    // representation
    event_pyclass.def(
        "__repr__",
        [](cuda::Event & event) { return event.str(); }
    );
}

// Wrap merlin::cuda library
void wrap_cuda(py::module & merlin_package) {
    // add cuda submodule
    py::module cuda_module = merlin_package.def_submodule("cuda", "CUDA runtime API wrapper.");
    // add classes and enums
    wrap_enums(cuda_module);
    wrap_device(cuda_module);
    wrap_event(cuda_module);
}

}  // namespace merlin
