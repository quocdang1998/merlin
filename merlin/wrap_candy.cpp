// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/candy/model.hpp"  // merlin::candy::Model


namespace merlin {

// Wrap merlin::candy::Model class
void wrap_model(py::module & candy_module) {
    auto model_pyclass = py::class_<candy::Model>(
        candy_module,
        "Model",
        R"(
        Canonical decomposition model.

        Wrapper of :cpp:class:`merlin::candy::Model`.)"
    );
    // constructor
    model_pyclass.def(
        py::init(
            [](py::sequence & shape, std::uint64_t rank) {
                return new candy::Model(pyseq_to_array<std::uint64_t>(shape), rank);
            }
        ),
        R"(
        Constructor from train data shape and rank.
        
        Parameters
        ----------
        shape : Sequence[int]
            Shape of decompressed data.
        rank : int
            Rank of canonical decomposition model (number of vector per axis).)",
        py::arg("shape"), py::arg("rank")
    );
    // attributes
    model_pyclass.def_property_readonly(
        "ndim",
        [](const candy::Model & self) { return self.ndim(); },
        "Get number of dimension."
    );
    model_pyclass.def_property_readonly(
        "rshape",
        [](const candy::Model & self) { return py::cast(self.rshape()); },
        "Get rank by shape."
    );
    model_pyclass.def_property_readonly(
        "rank",
        [](const candy::Model & self) { return self.rank(); },
        "Get rank."
    );
    model_pyclass.def_property_readonly(
        "num_params",
        [](const candy::Model & self) { return self.num_params(); },
        "Get number of parameters."
    );
    // Get and set parameters
    model_pyclass.def(
        "get",
        [](const candy::Model & self, std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank) {
            return self.get(i_dim, index, rank);
        },
        "Get an element at a given dimension, index and rank.",
        py::arg("i_dim"), py::arg("index"), py::arg("rank")
    );
    model_pyclass.def(
        "get",
        [](const candy::Model & self, std::uint64_t c_index) { return self[c_index]; },
        "Get an element from flattened index.",
        py::arg("c_index")
    );
    model_pyclass.def(
        "set",
        [](candy::Model & self, std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank, double value) {
            self.get(i_dim, index, rank) = value;
        },
        "Set value to an element at a given dimension, index and rank.",
        py::arg("i_dim"), py::arg("index"), py::arg("rank"), py::arg("value")
    );
    model_pyclass.def(
        "set",
        [](candy::Model & self, std::uint64_t c_index, double value) { self[c_index] = value; },
        "Set an element from flattened index.",
        py::arg("c_index"), py::arg("value")
    );
    // evaluation
    model_pyclass.def(
        "eval",
        [](candy::Model & self, py::sequence & index) {
            return self.eval(pyseq_to_array<std::uint64_t>(index));
        },
        "Evaluate result of the model at a given ndim index in the resulted array.",
        py::arg("index")
    );
    // check negative
    model_pyclass.def(
        "check_negative",
        [](candy::Model & self) { return self.check_negative(); },
        "Check if these is a negative parameter in the model."
    );
    // initialization
    model_pyclass.def(
        "initialize",
        [](candy::Model & self, const array::Array & train_data) {
            self.initialize(train_data);
        },
        "Initialize values of model based on train data.",
        py::arg("train_data")
    );
    // serialization
    model_pyclass.def(
        "save",
        [](candy::Model & self, const std::string & fname, bool lock) { self.save(fname, lock); },
        R"(
        Write model into a file.
        
        Parameters
        ----------
        fname : str
            Name of the output file.
        lock : bool, default=False
            Lock the file when writing to prevent data race. The lock action may cause a delay.)",
        py::arg("fname"), py::arg("lock") = false
    );
    // representation
    model_pyclass.def(
        "__repr__",
        [](const candy::Model & self) { return self.str(); }
    );
}

// Create empty C-contiguous array
static void wrap_load_model(py::module & candy_module) {
    // load model from a file
    candy_module.def(
        "load_model",
        [](const std::string & fname, bool lock) {
            candy::Model * p_model = new candy::Model();
            p_model->load(fname, lock);
            return p_model;
        },
        R"(
        Read model from a file.

        Parameters
        ----------
        fname : str
            Name of the input file.
        lock : bool, default=False
            Lock the file when writing to prevent data race. The lock action may cause a delay.)",
        py::arg("fname"), py::arg("lock") = false
    );
}


void wrap_candy(py::module & merlin_package) {
    // add candy submodule
    py::module candy_module = merlin_package.def_submodule("candy", "Data compression by Candecomp-Paraface method.");
    // add classes
    wrap_model(candy_module);
    wrap_load_model(candy_module);
}

}  // namespace merlin
