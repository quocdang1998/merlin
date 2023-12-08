// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <vector>  // std::vector

#include "merlin/array/array.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace merlin {

// Wrap merlin::candy::Model class
void wrap_model(py::module & candy_module) {
    auto model_pyclass = py::class_<candy::Model>(
        candy_module,
        "Model",
        R"(
        Canonical decomposition model.

        Wrapper of :cpp:class:`merlin::candy::Model`.
        )"
    );
    // constructors
    model_pyclass.def(
        py::init([]() { return new candy::Model(); }),
        "Default constructor."
    );
    model_pyclass.def(
        py::init(
            [](py::list & shape, std::uint64_t rank) {
                std::vector<std::uint64_t> shape_cpp = shape.cast<std::vector<std::uint64_t>>();
                intvec shape_merlin;
                shape_merlin.assign(shape_cpp.data(), shape_cpp.size());
                return new candy::Model(shape_merlin, rank);
            }
        ),
        "Constructor from train data shape and rank.",
        py::arg("shape"), py::arg("rank")
    );
    model_pyclass.def(
        py::init(
            [](py::list & param_vectors, std::uint64_t rank) {
                std::vector<std::vector<double>> param_vector_cpp;
                param_vector_cpp.reserve(param_vectors.size());
                for (auto it : param_vectors) {
                    param_vector_cpp.push_back((*it).cast<std::vector<double>>());
                }
                Vector<floatvec> param_vector_merlin(param_vector_cpp.size());
                for(std::uint64_t i = 0; i < param_vector_merlin.size(); i++) {
                    param_vector_merlin[i].assign(param_vector_cpp[i].data(), param_vector_cpp[i].size());
                }
                return new candy::Model(param_vector_merlin, rank);
            }
        ),
        "Constructor from model values.",
        py::arg("param_vectors"), py::arg("rank")
    );
    // attributes
    model_pyclass.def_property_readonly(
        "ndim",
        [](const candy::Model & self) { return self.ndim(); },
        "Get number of dimension."
    );
    model_pyclass.def_property_readonly(
        "rshape",
        [](const candy::Model & self) {
            const intvec & rshape = self.rshape();
            std::vector rshape_cpp(rshape.cbegin(), rshape.cend());
            py::list rshape_python = py::cast(rshape_cpp);
            return rshape_python;
        },
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
            std::vector<std::uint64_t> index_cpp = index.cast<std::vector<std::uint64_t>>();
            intvec index_merlin;
            index_merlin.assign(index_cpp.data(), index_cpp.size());
            return self.eval(index_merlin);
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
        [](candy::Model & self, const array::Array & train_data, std::uint64_t n_thread) {
            self.initialize(train_data, n_thread);
        },
        "Initialize values of model based on train data.",
        py::arg("train_data"), py::arg("n_thread") = 1
    );
    // representation
    model_pyclass.def(
        "__repr__",
        [](const candy::Model & self) { return self.str(); }
    );
}

void wrap_candy(py::module & merlin_package) {
    // add candy submodule
    py::module candy_module = merlin_package.def_submodule("candy", "Data compression by Candecomp-Paraface method.");
    // add classes
    wrap_model(candy_module);
}

}  // namespace merlin
