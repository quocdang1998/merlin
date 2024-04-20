// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include <cstddef>        // nullptr

#include "merlin/array/array.hpp"        // merlin::array::Array
#include "merlin/array/parcel.hpp"       // merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::TrainMetric
#include "merlin/candy/gradient.hpp"     // merlin::candy::Gradient
#include "merlin/candy/model.hpp"        // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"    // merlin::candy::Optimizer
#include "merlin/candy/trainer.hpp"      // merlin::candy::Trainer

namespace merlin {

// wrap TrainMetric
static std::map<std::string, candy::TrainMetric> trainmetric_map = {
    {"relsquare", candy::TrainMetric::RelativeSquare},
    {"abssquare", candy::TrainMetric::AbsoluteSquare}
};

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
        [](const candy::Model & self) { return array_to_pylist(self.rshape(), self.ndim()); },
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

// Wrap merlin::candy::Gradient class
void wrap_gradient(py::module & candy_module) {
    auto gradient_pyclass = py::class_<candy::Gradient>(
        candy_module,
        "Gradient",
        R"(
        Gradient tape for the canonical decomposition model.

        Wrapper of :cpp:class:`merlin::candy::Gradient`.

        Examples
        --------
        >>> with Gradient(model.num_params) as g:
        >>>     g.calc(model, data)
        >>>     print(g)
        )"
    );
    // constructor
    gradient_pyclass.def(
        py::init(
            [](std::uint64_t num_params, const std::string & metric) {
                return new candy::Gradient(nullptr, num_params, trainmetric_map.at(metric));
            }
        ),
        R"(
        Construct an empty gradient tape with size.

        Notes
        -----
        The gradient when constructed alone is not valid. A valid Gradient must be constructed by the ``with`` syntax:

        Parameters
        ----------
        num_params : int
            Number of parameters of the target model.
        metric : str, default="relsquare"
            Loss function to calculate the gradient.)",
        py::arg("num_params"), py::arg("metric") = "relsquare"
    );
    // enter and exit
    gradient_pyclass.def(
        "__enter__",
        [](candy::Gradient & self) {
            double * gradient_memory = new double[self.value().size()];
            self.value().data() = gradient_memory;
            return &self;
        },
        R"(
        Allocate memory and make the gradient valid.)"
    );
    gradient_pyclass.def(
        "__exit__",
        [](candy::Gradient & self, py::object type, py::object value, py::object traceback) {
            delete[] self.value().data();
            self.value().data() = nullptr;
        },
        R"(
        De-allocate memory and make the gradient invalid.)"
    );
    // calculate gradient
    gradient_pyclass.def(
        "calc",
        [](candy::Gradient & self, candy::Model & model, const array::Array & data) {
            Index index;
            self.calc_by_cpu(model, data, 0, 1, index);
        },
        R"(
        Calculate the gradient of a model wrt. a data.

        If the shape of 2 objects are not coincide, this function will result in undefined behavior.

        Parameters
        ----------
        model : merlin.candy.Model
            Target model to calculate the gradient.
        data : merlin.array.Array
            Data to calculate the gradient.)",
        py::arg("model"), py::arg("data")
    );
    // get Python memory view to the result of the gradient
    gradient_pyclass.def(
        "value",
        [](candy::Gradient & self) {
            return py::memoryview::from_buffer(self.value().data(), {self.value().size()}, {sizeof(double)});
        },
        R"(
        Get a Python memory view to the gradient calculated.

        This memory view can be transformed into list or Numpy array as demanded by user.)"
    );
    // representation
    gradient_pyclass.def(
        "__repr__",
        [](const candy::Gradient & self) { return self.str(); }
    );
}

// Wrap merlin::candy::Optimizer class
void wrap_optimizer(py::module & candy_module) {
    auto optimizer_pyclass = py::class_<candy::Optimizer>(
        candy_module,
        "Optimizer",
        R"(
        Algorithm for updating a model based on its gradient.

        Wrapper of :cpp:class:`merlin::candy::Optimizer`.)"
    );
    // constructor
    optimizer_pyclass.def(
        py::init(
            [](void) {
                return new candy::Optimizer();
            }
        ),
        R"(
        Default constructor.)"
    );
    // create grad descent
    candy_module.def(
        "create_grad_descent",
        [](double learning_rate) {
            return new candy::Optimizer(candy::create_grad_descent(learning_rate));
        },
        R"(
        Create an optimizer with gradient descent algorithm.

        See also :cpp:class:`merlin::candy::optmz::GradDescent`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.)",
        py::arg("learning_rate")
    );
    // create adagrad
    candy_module.def(
        "create_adagrad",
        [](double learning_rate, const candy::Model & model, double bias) {
            return new candy::Optimizer(candy::create_adagrad(learning_rate, model, bias));
        },
        R"(
        Create an optimizer with AdaGrad algorithm.

        See also :cpp:class:`merlin::candy::optmz::AdaGrad`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.
        model : merlin.candy.Model
            Model to fit.
        bias : float, default=1e-8
            Bias.)",
        py::arg("learning_rate"), py::arg("model"), py::arg("bias") = 1e-8
    );
    // create adam
    candy_module.def(
        "create_adam",
        [](double learning_rate, double beta_m, double beta_v, const candy::Model & model, double bias) {
            return new candy::Optimizer(candy::create_adam(learning_rate, beta_m, beta_v, model, bias));
        },
        R"(
        Create an optimizer with ADAM algorithm.

        See also :cpp:class:`merlin::candy::optmz::Adam`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.
        beta_m : float
            First moment decay constant.
        beta_v : float
            Second moment decay constant.
        model : merlin.candy.Model
            Model to fit.
        bias : float, default=1e-8
            Bias.)",
        py::arg("learning_rate"), py::arg("beta_m"), py::arg("beta_v"), py::arg("model"), py::arg("bias") = 1e-8
    );
    // create adadelta
    candy_module.def(
        "create_adadelta",
        [](double learning_rate, double decay_constant, const candy::Model & model, double bias) {
            return new candy::Optimizer(candy::create_adadelta(learning_rate, decay_constant, model, bias));
        },
        R"(
        Create an optimizer with AdaDelta algorithm.

        See also :cpp:class:`merlin::candy::optmz::AdaDelta`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.
        decay_constant : float
            Decay constant.
        model : merlin.candy.Model
            Model to fit.
        bias : float, default=1e-8
            Bias.)",
        py::arg("learning_rate"), py::arg("decay_constant"), py::arg("model"), py::arg("bias") = 1e-8
    );
}

// Wrap merlin::candy::Trainer class
void wrap_trainer(py::module & candy_module) {
    auto trainer_pyclass = py::class_<candy::Trainer>(
        candy_module,
        "Trainer",
        R"(
        Launch a train process on Candecomp model asynchronously.

        Wrapper of :cpp:class:`merlin::candy::Trainer`.)"
    );
    // constructor
    trainer_pyclass.def(
        py::init(
            [](const std::string & name, const candy::Model & model, const candy::Optimizer & optimizer,
               Synchronizer & synch) {
                return new candy::Trainer(name, model, optimizer, synch);
            }
        ),
        R"(
        Construct a trainer from model and optimizer.

        Parameters
        ----------
        name : str
            Name of the trainer.
        model : merlin.candy.Model
            Model to train.
        optimizer : merlin.candy.Optimizer
            Optimizer training the model.
        synch : merlin.Synchronizer
            Asynchronous stream to register the training process. Destroying the synchronizer before the Trainer results
            in undefined behavior.)",
        py::arg("name"), py::arg("model"), py::arg("optimizer"), py::arg("synch"), py::keep_alive<1,5>()
    );
    // attributes
    trainer_pyclass.def_property_readonly(
        "model",
        [](const candy::Trainer & self) { return &(self.model()); },
        "Get the current model.",
        py::keep_alive<0, 1>()
    );
    trainer_pyclass.def_property_readonly(
        "optmz",
        [](const candy::Trainer & self) { return &(self.optmz()); },
        "Get the current optimizer.",
        py::keep_alive<0, 1>()
    );
    // change optimizer
    trainer_pyclass.def(
        "change_optmz",
        [](candy::Trainer & self, candy::Optimizer * p_new_optmz) { self.change_optmz(std::move(*p_new_optmz)); },
        "Change optimizer."
    );
    // train model
    trainer_pyclass.def(
        "update_cpu",
        [](candy::Trainer & self, const array::Array & data, std::uint64_t rep, double threshold,
           std::uint64_t n_threads, const std::string & metric) {
            self.update(data, rep, threshold, n_threads, trainmetric_map.at(metric));
        },
        R"(
        Update CP model according to gradient on CPU.

        Update CP model for a certain number of iterations, and check if the relative error after the training
        process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
        again and check.

        Parameters
        ----------
        data : merlin.array.Array
            Data to train the model.
        rep : int
            Number of times to repeat the gradient descent update in each step.
        threshold : float
            Threshold to stop the training process.
        n_threads : int, default=1
            Number of parallel threads for training the model.
        metric : str, default="relsquare"
            Training metric for the model.)",
        py::arg("data"), py::arg("rep"), py::arg("threshold"), py::arg("n_threads") = 1, py::arg("metric") = "relsquare"
    );
    trainer_pyclass.def(
        "update_gpu",
        [](candy::Trainer & self, const array::Parcel & data, std::uint64_t rep, double threshold,
           std::uint64_t n_threads, const std::string & metric) {
            self.update(data, rep, threshold, n_threads, trainmetric_map.at(metric));
        },
        R"(
        Update CP model according to gradient on GPU.

        Update CP model for a certain number of iterations, and check if the relative error after the training
        process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
        again and check.

        Parameters
        ----------
        data : merlin.array.Parcel
            Data to train the model.
        rep : int
            Number of times to repeat the gradient descent update in each step.
        threshold : float
            Threshold to stop the training process.
        n_threads : int, default=32
            Number of parallel threads for training the model.
        metric : str, default="relsquare"
            Training metric for the model.)",
        py::arg("data"), py::arg("rep"), py::arg("threshold"), py::arg("n_threads") = 32,
        py::arg("metric") = "relsquare"
    );
    // calculate error
    trainer_pyclass.def(
        "error_cpu",
        [](candy::Trainer & self, const array::Array & data, std::uint64_t n_threads) {
            double * errors = new double[2];
            self.get_error(data, errors[0], errors[1], n_threads);
            return make_wrapper_array<double>(errors, 2);
        },
        R"(
        Asynchronous calculate error.

        Get the RMSE and RMAE error with respect to a given dataset by CPU.

        Parameters
        ----------
        data : merlin.array.Array
            Data to train the model.
        n_threads : int, default=1
            Number of parallel threads for training the model.
        )",
        py::arg("data"), py::arg("n_threads") = 1
    );
    trainer_pyclass.def(
        "error_gpu",
        [](candy::Trainer & self, const array::Parcel & data, std::uint64_t n_threads) {
            double * errors = new double[2];
            self.get_error(data, errors[0], errors[1], n_threads);
            return make_wrapper_array<double>(errors, 2);
        },
        R"(
        Asynchronous calculate error.

        Get the RMSE and RMAE error with respect to a given dataset by GPU.

        Parameters
        ----------
        data : merlin.array.Parcel
            Data to train the model.
        n_threads : int, default=32
            Number of parallel threads for training the model.
        )",
        py::arg("data"), py::arg("n_threads") = 32
    );
    // dry run
    trainer_pyclass.def(
        "dry_run_cpu",
        [](candy::Trainer & self, const array::Array & data, std::uint64_t max_iter, std::uint64_t n_threads,
           const std::string & metric) {
            DoubleVec error(max_iter);
            UIntVec count(1);
            self.dry_run(data, error, count[0], max_iter, n_threads, trainmetric_map.at(metric));
            double * error_data = std::exchange(error.data(), nullptr);
            std::uint64_t * count_data = std::exchange(count.data(), nullptr);
            return std::make_pair(make_wrapper_array<double>(error_data, error.size()),
                                  make_wrapper_array<std::uint64_t>(count_data, 1));
        },
        R"(
        Dry-run using CPU parallelism.

        Perform gradient descent algorithm asynchronously for a given number of iterations without updating the model.
        The iterative process will stop when the RMSE after updated is bigger than before.

        Parameters
        ----------
        data : merlin.array.Array
            Data to train the model.
        max_iter: int, default=100
            Max number of iterations of the dry-run.
        n_threads : int, default=1
            Number of parallel threads for calculating the gradient.
        metric : str, default="relsquare"
            Training metric for the model.
        Returns
        -------
        error : np.array
            Vector storing error per iteration.
        count : np.array
            Array of size 1, storing the number of iterations performed before breaking the dry run.
        )",
        py::arg("data"), py::arg("max_iter") = 100, py::arg("n_threads") = 1, py::arg("metric") = "relsquare"
    );
    trainer_pyclass.def(
        "dry_run_gpu",
        [](candy::Trainer & self, const array::Parcel & data, std::uint64_t max_iter, std::uint64_t n_threads,
           const std::string & metric) {
            DoubleVec error(max_iter);
            UIntVec count(1);
            self.dry_run(data, error, count[0], max_iter, n_threads, trainmetric_map.at(metric));
            double * error_data = std::exchange(error.data(), nullptr);
            std::uint64_t * count_data = std::exchange(count.data(), nullptr);
            return std::make_pair(make_wrapper_array<double>(error_data, error.size()),
                                  make_wrapper_array<std::uint64_t>(count_data, 1));
        },
        R"(
        Dry-run using GPU parallelism.

        Perform gradient descent algorithm asynchronously for a given number of iterations without updating the model.
        The iterative process will stop when the RMSE after updated is bigger than before.

        Parameters
        ----------
        data : merlin.array.Parcel
            Data to train the model.
        max_iter: int, default=100
            Max number of iterations of the dry-run.
        n_threads : int, default=32
            Number of parallel threads for calculating the gradient.
        metric : str, default="relsquare"
            Training metric for the model.
        Returns
        -------
        error : np.array
            Vector storing error per iteration.
        count : np.array
            Array of size 1, storing the number of iterations performed before breaking the dry run.
        )",
        py::arg("data"), py::arg("max_iter") = 100, py::arg("n_threads") = 32, py::arg("metric") = "relsquare"
    );
    // reconstruct
    trainer_pyclass.def(
        "reconstruct_cpu",
        [](candy::Trainer & self, array::Array & dest, std::uint64_t n_threads) { self.reconstruct(dest, n_threads); },
        R"(
        Reconstruct a whole multi-dimensional data from the model using CPU parallelism. 

        Parameters
        ----------
        dest : merlin.array.Array
            Array to write result.
        n_threads : int, default=1
            Number of parallel threads for training the model.
        )",
        py::arg("dest"), py::arg("n_threads") = 1
    );
    trainer_pyclass.def(
        "reconstruct_gpu",
        [](candy::Trainer & self, array::Parcel & dest, std::uint64_t n_threads) { self.reconstruct(dest, n_threads); },
        R"(
        Reconstruct a whole multi-dimensional data from the model using GPU parallelism. 

        Parameters
        ----------
        dest : merlin.array.Parcel
            Array to write result.
        n_threads : int, default=32
            Number of parallel threads for training the model.
        )",
        py::arg("dest"), py::arg("n_threads") = 32
    );
}

void wrap_candy(py::module & merlin_package) {
    // add candy submodule
    py::module candy_module = merlin_package.def_submodule("candy", "Data compression by Candecomp-Paraface method.");
    // add classes
    wrap_model(candy_module);
    wrap_gradient(candy_module);
    wrap_optimizer(candy_module);
    wrap_trainer(candy_module);
}

}  // namespace merlin
