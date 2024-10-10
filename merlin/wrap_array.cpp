// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include "merlin/array/array.hpp"   // merlin::array::Array
#include "merlin/array/nddata.hpp"  // merlin::array::NdData
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/array/stock.hpp"   // merlin::array::Stock

namespace merlin {

// Wrap merlin::array::NdData class
static void wrap_nddata(py::module & array_module) {
    auto nddata_pyclass = py::class_<array::NdData>(
        array_module,
        "NdData",
        R"(
        Abstract class of N-dim array.

        Wrapper of :cpp:class:`merlin::array::NdData`.
        )"
    );
    // constructor
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
        [](const array::NdData & self) { return array_to_pylist(self.shape()); },
        "Get shape vector."
    );
    nddata_pyclass.def_property_readonly(
        "strides",
        [](const array::NdData & self) { return array_to_pylist(self.strides()); },
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
        [](const array::NdData & self, py::tuple & index) { return self.get(pyseq_to_array<std::uint64_t>(index)); },
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
            self.set(pyseq_to_array<std::uint64_t>(index), value);
        },
        "Set value of element at a ndim index.",
        py::arg("index"), py::arg("value")
    );
    // operations
    nddata_pyclass.def(
        "reshape",
        [](array::NdData & self, py::sequence & new_shape) { self.reshape(pyseq_to_array<std::uint64_t>(new_shape)); },
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

// Wrap merlin::array::Array class
static void wrap_array_(py::module & array_module) {
    auto array_pyclass = py::class_<array::Array, array::NdData>(
        array_module,
        "Array",
        py::buffer_protocol(),
        R"(
        Multi-dimensional array on CPU.

        Wrapper of :cpp:class:`merlin::array::Array`.
        )"
    );
    // constructor
    array_pyclass.def(
        py::init(
            [](py::buffer buffer, bool copy, bool pin_memory) {
                py::buffer_info info = buffer.request();
                if (info.format != py::format_descriptor<double>::format()) {
                    throw std::runtime_error("Incompatible format: expected a double array.");
                }
                std::uint64_t ndim(info.ndim);
                Index shape(info.ndim), strides(info.ndim);
                for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
                    shape[i_dim] = std::uint64_t(info.shape[i_dim]);
                    strides[i_dim] = std::uint64_t(info.strides[i_dim]);
                }
                return new array::Array(reinterpret_cast<double *>(info.ptr), shape, strides, copy, pin_memory);
            }
        ),
        R"(
        Construct array from buffer memory (Numpy array, Pandas array, etc).

        Parameters
        ----------
        buffer :
            Original array.
        copy : bool
            If ``True``, copy data from the buffer to a new C-contiguous array. otherwise, directly assign the array
            to the pointer of the buffer.
        pin_memory : bool
            If ``True``, pin pages containing assigned data to the memory. Pinned memory pages cannot be swapped out of
            the memory. Memory should only be pages once, and the paging order must be performed on the assigned array
            containing the first element of the whole array.)",
        py::arg("buffer"), py::arg("copy") = false, py::arg("pin_memory") = true,
        py::keep_alive<1,2>()
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

// Wrap merlin::array::Parcel class
static void wrap_parcel(py::module & array_module) {
    auto parcel_pyclass = py::class_<array::Parcel, array::NdData>(
        array_module,
        "Parcel",
        R"(
        Multi-dimensional array on GPU.

        Wrapper of :cpp:class:`merlin::array::Parcel`.
        )"
    );
    // constructor
    parcel_pyclass.def(
        py::init([]() { return new array::Parcel(); }),
        "Default constructor."
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
    // deallocate current data
    parcel_pyclass.def(
        "free_current_data",
        [](array::Parcel & self, const cuda::Stream & stream) {
            self.free_current_data(stream);
        },
        "Deallocate the momory on GPU pointed by the object.",
        py::arg("stream") = cuda::Stream()
    );
}

// Wrap merlin::array::Stock class
void wrap_stock(py::module & array_module) {
    auto stock_pyclass = py::class_<array::Stock, array::NdData>(
        array_module,
        "Stock",
        R"(
        Multi-dimensional array exported to a file.

        Wrapper of :cpp:class:`merlin::array::Stock`.
        )"
    );
    // constructor
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

// Create empty C-contiguous array
static void wrap_empty_constructors(py::module & array_module) {
    // empty CPU array
    array_module.def(
        "empty_array",
        [](py::sequence & shape) {
            return new array::Array(pyseq_to_array<std::uint64_t>(shape));
        },
        R"(
        Construct C-contiguous empty CPU array from shape vector.

        Parameters
        ----------
        shape : Sequence[int]
            Shape of n-dimensional array.)",
        py::arg("shape")
    );
    // empty GPU array
    array_module.def(
        "empty_parcel",
        [](py::sequence & shape, const cuda::Stream & stream) {
            return new array::Parcel(pyseq_to_array<std::uint64_t>(shape), stream);
        },
        R"(
        Construct C-contiguous empty GPU array from shape vector.

        Parameters
        ----------
        shape : Sequence[int]
            Shape of n-dimensional array.
        stream : merlin.cuda.Stream
            Asynchronous stream for asynchronous memory allocation.)",
        py::arg("shape"), py::arg("stream") = cuda::Stream()
    );
    // empty out-of-core array
    array_module.def(
        "empty_stock",
        [](const std::string & filename, py::sequence & shape, std::uint64_t offset, bool thread_safe) {
            return new array::Stock(filename, pyseq_to_array<std::uint64_t>(shape), offset, thread_safe);
        },
        R"(
        Open an empty file for storing data.

        Parameters
        ----------
        filename : str
            Name of the file storing data of the array.
        shape : Sequence[int]
            Shape of n-dimensional array.
        offset : int, default=0
            Number of bytes to start the reading, counting from the beginning of the file.
        thread_safe : bool, default=False
            Using filelock to avoid data race when the file is accessed by multiple threads.)",
        py::arg("filename"), py::arg("shape"), py::arg("offset") = 0, py::arg("thread_safe") = false
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
    wrap_empty_constructors(array_module);
}

}  // namespace merlin

