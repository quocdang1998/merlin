// Copyright 2023 quocdang1998
#include "merlin/array/array.hpp"
#include "merlin/array/nddata.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/array/stock.hpp"

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace merlin {

// Wrap merlin::NdData class
void wrap_nddata(py::module & array_module) {
    auto nddata_pyclass = py::class_<array::NdData>(
        array_module,
        "NdData",
        R"(
        Abstract class of N-dim array.

        Wrapper of :cpp:class:`merlin::array::NdData`.
        )"
    );
    // constructors
    nddata_pyclass.def(
        py::init([]() { return new array::NdData(); }),
        "Default constructor."
    );
    // get attributes
    nddata_pyclass.def_property_readonly(
        "ndim",
        [](const array::NdData & self) { return self.ndim(); },
        "Get number of dimension."
    );
    nddata_pyclass.def_property_readonly(
        "shape",
        [](const array::NdData & self) {
            const intvec & shape = self.shape();
            std::vector shape_cpp(shape.cbegin(), shape.cend());
            py::list shape_python = py::cast(shape_cpp);
            return shape_python;
        },
        "Get shape vector."
    );
    nddata_pyclass.def_property_readonly(
        "strides",
        [](const array::NdData & self) {
            const intvec & strides = self.strides();
            std::vector strides_cpp(strides.cbegin(), strides.cend());
            py::list strides_python = py::cast(strides_cpp);
            return strides_python;
        },
        "Get stride vector."
    );
    nddata_pyclass.def_property_readonly(
        "size",
        [](const array::NdData & self) { return self.size(); },
        "Get number of element."
    );
    nddata_pyclass.def_property_readonly(
        "is_c_contiguous",
        [](const array::NdData & self) { return self.is_c_contiguous(); },
        "Check if the array is C-contiguous."
    );
    // get / set elements
    nddata_pyclass.def(
        "get",
        [](const array::NdData & self, std::uint64_t c_index) { return self.get(c_index); },
        "Get value of element at a C-contiguous index.",
        py::arg("c_index")
    );
    nddata_pyclass.def(
        "get",
        [](const array::NdData & self, py::tuple & index) {
            std::vector<std::uint64_t> index_cpp = index.cast<std::vector<std::uint64_t>>();
            intvec index_merlin;
            index_merlin.assign(index_cpp.data(), index_cpp.size());
            return self.get(index_merlin);
        },
        "Get value of element at a n-dim index.",
        py::arg("index")
    );
    nddata_pyclass.def(
        "set",
        [](array::NdData & self, std::uint64_t c_index, double value) { return self.set(c_index, value); },
        "Set value of element at a C-contiguous index.",
        py::arg("c_index"), py::arg("value")
    );
    nddata_pyclass.def(
        "set",
        [](array::NdData & self, py::tuple & index, double value) {
            std::vector<std::uint64_t> index_cpp = index.cast<std::vector<std::uint64_t>>();
            intvec index_merlin;
            index_merlin.assign(index_cpp.data(), index_cpp.size());
            self.set(index_merlin, value);
        },
        "Set value of element at a ndim index.",
        py::arg("index"), py::arg("value")
    );
    // operations
    nddata_pyclass.def(
        "reshape",
        [](array::NdData & self, py::sequence & new_shape) {
            std::vector<std::uint64_t> new_shape_cpp = new_shape.cast<std::vector<std::uint64_t>>();
            self.reshape(intvec(new_shape_cpp.data(), new_shape_cpp.size()));
        },
        "Reshape the dataset.",
        py::arg("new_shape")
    );
    nddata_pyclass.def(
        "squeeze",
        [](array::NdData & self) { self.squeeze(); },
        "Collapse all dimensions with size 1."
    );
    nddata_pyclass.def(
        "fill",
        [](array::NdData & self, double value) { self.fill(value); },
        "Set value of all elements.",
        py::arg("value")
    );
    // representation
    nddata_pyclass.def(
        "__repr__",
        [](const array::NdData & self) { return self.str(); }
    );
}

// Wrap merlin::Array class
void wrap_array_(py::module & array_module) {
    auto array_pyclass = py::class_<array::Array, array::NdData>(
        array_module,
        "Array",
        py::buffer_protocol(),
        R"(
        Multi-dimensional array on CPU.

        Wrapper of :cpp:class:`merlin::array::Array`.
        )"
    );
    // constructors
    array_pyclass.def(
        py::init([]() { return new array::Array(); }),
        "Default constructor."
    );
    array_pyclass.def(
        py::init(
            [](py::list & shape) {
                std::vector<std::uint64_t> shape_cpp = shape.cast<std::vector<std::uint64_t>>();
                return new array::Array(intvec(shape_cpp.data(), shape_cpp.size()));
            }
        ),
        "Construct C-contiguous empty array from dimension vector.",
        py::arg("shape")
    );
    array_pyclass.def(
        py::init(
            [](py::buffer buffer, bool copy) {
                py::buffer_info info = buffer.request();
                if (info.format != py::format_descriptor<double>::format()) {
                    throw std::runtime_error("Incompatible format: expected a double array.");
                }
                std::uint64_t ndim(info.ndim);
                merlin::intvec shape(info.ndim), strides(info.ndim);
                for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
                    shape[i_dim] = std::uint64_t(info.shape[i_dim]);
                    strides[i_dim] = std::uint64_t(info.strides[i_dim]);
                }
                return new merlin::array::Array(reinterpret_cast<double *>(info.ptr), shape, strides, copy);
            }
        ),
        "Construct array from pointer, to data and meta-data.",
        py::arg("buffer"), py::arg("copy") = false
    );
    // conversion to Numpy
    array_pyclass.def_buffer(
        [](array::Array & self) {
            return py::buffer_info(self.data(), sizeof(double), py::format_descriptor<double>::format(), self.ndim(),
                                   self.shape(), self.strides());
        }
    );
    // copy data from GPU
    array_pyclass.def(
        "clone_data_from_gpu",
        [](array::Array & self, const array::Parcel & src, const cuda::Stream & stream) {
            self.clone_data_from_gpu(src, stream);
        },
        "Copy data from GPU array.",
        py::arg("src"), py::arg("stream") = cuda::Stream()
    );
    // read data from file
    array_pyclass.def(
        "extract_data_from_file",
        [](array::Array & self, const array::Stock & src) { self.extract_data_from_file(src); },
        "Export data to a file.",
        py::arg("src")
    );
}

// Wrap merlin::Parcel class
void wrap_parcel(py::module & array_module) {
    auto parcel_pyclass = py::class_<array::Parcel, array::NdData>(
        array_module,
        "Parcel",
        R"(
        Multi-dimensional array on GPU.

        Wrapper of :cpp:class:`merlin::array::Parcel`.
        )"
    );
    // constructors
    parcel_pyclass.def(
        py::init([]() { return new array::Parcel(); }),
        "Default constructor."
    );
    parcel_pyclass.def(
        py::init(
            [](py::list & shape, const cuda::Stream & stream) {
                std::vector<std::uint64_t> shape_cpp = shape.cast<std::vector<std::uint64_t>>();
                return new array::Parcel(intvec(shape_cpp.data(), shape_cpp.size()), stream);
            }
        ),
        R"(
        __init__(self: merlin.array.Parcel, shape: list, stream: merlin.cuda.Stream = merlin.cuda.Stream()) -> None
        
        Construct a contiguous array from shape on GPU.
        )",
        py::arg("shape"), py::arg("stream") = cuda::Stream()
    );
    // transfer data to GPU
    parcel_pyclass.def(
        "transfer_data_to_gpu",
        [](array::Parcel & self, const array::Array & cpu_array, const cuda::Stream & stream) {
            self.transfer_data_to_gpu(cpu_array, stream);
        },
        "Transfer data to GPU from CPU array.",
        py::arg("cpu_array"), py::arg("stream") = cuda::Stream()
    );
}

// Wrap merlin::Stock class
void wrap_stock(py::module & array_module) {
    auto stock_pyclass = py::class_<array::Stock, array::NdData>(
        array_module,
        "Stock",
        R"(
        Multi-dimensional array exported to a file.

        Wrapper of :cpp:class:`merlin::array::Stock`.
        )"
    );
    // constructors
    stock_pyclass.def(
        py::init([]() { return new array::Stock(); }),
        "Default constructor."
    );
    stock_pyclass.def(
        py::init(
            [](const std::string & filename, py::list & shape, std::uint64_t offset, bool thread_safe) {
                std::vector<std::uint64_t> shape_cpp = shape.cast<std::vector<std::uint64_t>>();
                return new array::Stock(filename, intvec(shape_cpp.data(), shape_cpp.size()), offset, thread_safe);
            }
        ),
        "Open an empty file for storing data.",
        py::arg("filename"), py::arg("shape"), py::arg("offset") = 0, py::arg("thread_safe") = true
    );
    stock_pyclass.def(
        py::init(
            [](const std::string & filename, std::uint64_t offset, bool thread_safe) {
                return new array::Stock(filename, offset, thread_safe);
            }
        ),
        "Open an already existing file for reading and storing data.",
        py::arg("filename"), py::arg("offset") = 0, py::arg("thread_safe") = true
    );
    // get members
    stock_pyclass.def_property_readonly(
        "filename",
        [](const array::Stock & self) { return self.filename(); },
        "Get filename."
    );
    stock_pyclass.def_property_readonly(
        "is_thread_safe",
        [](const array::Stock & self) { return self.is_thread_safe(); },
        "Check if the policy is thread safe."
    );
    // write to file
    stock_pyclass.def(
        "record_data_to_file",
        [](array::Stock & self, const array::Array & src) { self.record_data_to_file(src); },
        "Write data from a CPU array to a file.",
        py::arg("src")
    );
}

void wrap_array(py::module & merlin_package) {
    // add array submodule
    py::module array_module = merlin_package.def_submodule("array", "Multi-dimensional array wrapper API.");
    // add classes
    wrap_nddata(array_module);
    wrap_array_(array_module);
    wrap_parcel(array_module);
    wrap_stock(array_module);
}

}  // namespace merlin
