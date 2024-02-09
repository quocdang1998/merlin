// Copyright 2023 quocdang1998
#include "merlin/cuda_interface.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/enum_wrapper.hpp"
#include "merlin/cuda/event.hpp"
#include "merlin/cuda/stream.hpp"

#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace merlin {

// wrap DeviceLimit
static std::unordered_map<std::string, cuda::DeviceLimit> devicelimit_map = {
    {"stacksize",          cuda::DeviceLimit::StackSize},
    {"printfsize",         cuda::DeviceLimit::PrintfSize},
    {"heapsize",           cuda::DeviceLimit::HeapSize},
    {"syncdepth",          cuda::DeviceLimit::SyncDepth},
    {"launchpendingcount", cuda::DeviceLimit::LaunchPendingCount}
};

// wrap StreamSetting
static std::unordered_map<std::string, cuda::StreamSetting> streamsetting_map = {
    {"default",     cuda::StreamSetting::Default},
    {"nonblocking", cuda::StreamSetting::NonBlocking}
};

// wrap EventWaitFlag
static std::unordered_map<std::string, cuda::EventWaitFlag> eventwaitflag_map = {
    {"default",  cuda::EventWaitFlag::Default},
    {"external", cuda::EventWaitFlag::External}
};

// Wrap enums
static void wrap_enums(py::module & cuda_module) {
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
        py::init([](int id){ return new cuda::Device(id); }),
        R"(
        Constructor from GPU ID.
        
        Parameters
        ----------
        id : int, default = -1
            ID of the GPU, ranging from 0 (included) to the number of GPUs attached to the machine (excluded). Input a
            negative number will result in an invalid GPU error.)",
        py::arg("id") = -1
    );
    // members
    device_pyclass.def_property(
        "id",
        [](cuda::Device & self) { return self.id(); },
        [](cuda::Device & self, int new_id) { self.id() = new_id; }
    );
    // query
    device_pyclass.def(
        "print_specification",
        [](cuda::Device & self) { self.print_specification(); },
        "Print GPU specifications."
    );
    device_pyclass.def(
        "test_gpu",
        [](cuda::Device & self) { return self.test_gpu(); },
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
        [](cuda::Device & self) { self.set_as_current(); },
        "Set the GPU as current device."
    );
    device_pyclass.def_static(
        "get_limit",
        [](const std::string & limit) { return cuda::Device::limit(devicelimit_map.at(limit)); },
        R"(
        Get setting limits of the current GPU.
        
        Parameters
        ----------
        limit : str
            Limit to get, possible values are ``stacksize``, ``printfsize``, ``heapsize``, ``syncdepth``,
            ``launchpendingcount``.)",
        py::arg("limit")
    );
    device_pyclass.def_static(
        "set_limit",
        [](const std::string & limit, std::uint64_t size) {
            return cuda::Device::limit(devicelimit_map.at(limit), size);
        },
        R"(
        Set setting limits of the current GPU.
        
        Parameters
        ----------
        limit : str
            Limit to set, possible values are ``stacksize``, ``printfsize``, ``heapsize``, ``syncdepth``,
            ``launchpendingcount``.)",
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
        [](cuda::Device & self) { return self.str(); }
    );
    // action on all GPUs
    cuda_module.def(
        "print_gpus_spec",
        &cuda::print_gpus_spec,
        R"(
        Print specification of all detected GPUs.

        Wrapper of :cpp:func:`merlin::cuda::print_gpus_spec`)"
    );
    cuda_module.def(
        "test_all_gpu",
        &cuda::test_all_gpu,
        R"(Perform a simple test on all detected GPU.

        Wrapper of :cpp:func:`merlin::cuda::test_all_gpu`)"
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
        py::init(
            [](bool blocking_sync, bool disable_timing, bool interprocess) {
                unsigned int category = cuda::EventCategory::Default;
                if (blocking_sync) {
                    category |= cuda::EventCategory::BlockingSync;
                }
                return new cuda::Event(category);
            }
        ),
        R"(
        Construct CUDA event from flag.
        
        Parameters
        ----------
        blocking_sync : bool, default = False
            Create a CUDA event with blocking synchronization flag. If this flag is on, the current CPU thread will be
            locked until the event occurs. Otherwise, it will be put in busy-waiting state (query and wait loop).
        disable_timing : bool, default = False
            Disable the capability to record time. Should be switched on if the event is only used for stream
            synchronization to improve performance of the code.
        interprocess : bool, default = False
            The event can be used for interprocess communication.)",
        py::arg("blocking_sync") = false, py::arg("disable_timing") = false, py::arg("interprocess") = false
    );
    // attributes
    event_pyclass.def(
        "get_event_ptr",
        [](cuda::Event & self) { return self.get_event_ptr(); },
        "Get pointer to CUDA event."
    );
    event_pyclass.def(
        "get_category",
        [](cuda::Event & self) { return self.get_category(); },
        "Get setting flag of the event."
    );
    event_pyclass.def(
        "get_gpu",
        [](cuda::Event & self) { return new cuda::Device(self.get_gpu()); },
        "Get GPU associated to the event."
    );
    // query
    event_pyclass.def(
        "is_complete",
        [](cuda::Event & self) { return self.is_complete(); },
        "Query the status of all work currently captured by event."
    );
    event_pyclass.def(
        "check_cuda_context",
        [](cuda::Event & self) { self.check_cuda_context(); },
        "Check validity of GPU and context."
    );
    // operation
    event_pyclass.def(
        "synchronize",
        [](cuda::Event & self) { self.synchronize(); },
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
        [](cuda::Event & self) { return self.str(); }
    );
}

// Wrap merlin::cuda::Event
static void wrap_stream(py::module & cuda_module) {
    auto stream_pyclass = py::class_<cuda::Stream>(
        cuda_module,
        "Stream",
        R"(
        CUDA stream.

        Wrapper of :cpp:class:`merlin::cuda::Stream`.)"
    );
    // constructor
    stream_pyclass.def(
        py::init([](const std::string & setting, int priority) {
            cuda::Stream * self;
            if (setting.compare("nullstream") == 0) {
                self = new cuda::Stream();
            } else {
                self = new cuda::Stream(streamsetting_map.at(setting), priority);
            }
            return self;
        }),
        R"(
        Construct CUDA stream from its setting and priority.
        
        Parameters
        ----------
        setting : str
            Setting of the CUDA stream. Can be either ``nullstream`` (assign to default null stream), ``default``
            (commands on the stream is blocked by the default null stream) or ``nonblocking`` (commands run parallel
            with those executed on the null stream).
        priority : int, default = 0
            Priority of the stream. Lower number means higher priority.)",
        py::arg("setting") = "nullstream", py::arg("priority") = 0
    );
    // attributes
    stream_pyclass.def(
        "get_stream_ptr",
        [](cuda::Stream & self) { return self.get_stream_ptr(); },
        "Get stream pointer."
    );
    stream_pyclass.def(
        "get_setting",
        [](cuda::Stream & self) {
            std::string setting;
            switch (self.get_setting()) {
                case cuda::StreamSetting::Default : {
                    setting = "default";
                }
                case cuda::StreamSetting::NonBlocking : {
                    setting = "nonblocking";
                }
            }
            return setting;
        },
        "Get setting flag of the stream."
    );
    stream_pyclass.def(
        "get_priority",
        [](cuda::Stream & self) { return self.get_priority(); },
        "Get priority of the stream."
    );
    stream_pyclass.def(
        "get_gpu",
        [](cuda::Stream & self) { return new cuda::Device(self.get_gpu()); },
        "Get GPU on which the stream resides."
    );
    // query
    stream_pyclass.def(
        "is_complete",
        [](cuda::Stream & self) { return self.is_complete(); },
        "Query for completion status."
    );
    stream_pyclass.def(
        "is_capturing",
        [](cuda::Stream & self) { return self.is_capturing(); },
        "Check if the stream is being captured."
    );
    stream_pyclass.def(
        "check_cuda_context",
        [](cuda::Stream & self) { self.check_cuda_context(); },
        "Check validity of GPU and context."
    );
    // operation
    stream_pyclass.def(
        "record_event",
        [](cuda::Stream & self, cuda::Event & event) { self.record_event(event); },
        "Register an event on the stream.",
        py::arg("event")
    );
    stream_pyclass.def(
        "wait_event",
        [](cuda::Stream & self, cuda::Event & event, const std::string & flag) {
            self.wait_event(event, eventwaitflag_map.at(flag));
        },
        R"(
        Make the stream wait on an event.
        
        Parameters
        ----------
        event : merlin.cuda.Event
            CUDA event to wait for.
        flag : str, default = "default"
            Wait flag, can be either ``default`` or ``external``. See :cpp:enum:`merlin::cuda::EventWaitFlag` for more
            information.
        )",
        py::arg("event"), py::arg("flag") = "default"
    );
    stream_pyclass.def(
        "synchronize",
        [](cuda::Stream & self) { self.synchronize(); },
        "Synchronize the stream."
    );
    // representation
    stream_pyclass.def(
        "__repr__",
        [](cuda::Stream & stream) { return stream.str(); }
    );
}

void wrap_cuda(py::module & merlin_package) {
    // add cuda submodule
    py::module cuda_module = merlin_package.def_submodule("cuda", "CUDA runtime API wrapper.");
    // add classes and enums
    wrap_enums(cuda_module);
    wrap_device(cuda_module);
    wrap_event(cuda_module);
    wrap_stream(cuda_module);
}

}  // namespace merlin
