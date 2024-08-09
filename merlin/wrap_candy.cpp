// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include <cstddef>        // nullptr

#include "merlin/array/array.hpp"         // merlin::array::Array
#include "merlin/array/parcel.hpp"        // merlin::array::Parcel
#include "merlin/candy/declaration.hpp"   // merlin::candy::TrainMetric
#include "merlin/candy/gradient.hpp"      // merlin::candy::Gradient
#include "merlin/candy/model.hpp"         // merlin::candy::Model
#include "merlin/candy/optimizer.hpp"     // merlin::candy::Optimizer
#include "merlin/candy/randomizer.hpp"    // merlin::candy::Randomizer
#include "merlin/candy/trainer.hpp"       // merlin::candy::Trainer
#include "merlin/candy/trial_policy.hpp"  // merlin::candy::TrialPolicy

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

        Wrapper of :cpp:class:`merlin::candy::Model`.
        )"
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
            Rank of canonical decomposition model (number of vector per axis).
        )",
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
        "all_positive",
        [](candy::Model & self) { return self.all_positive(); },
        "Check if all parameters in the model are positive."
    );
    // initialization
    model_pyclass.def(
        "initialize",
        [](candy::Model & self, const array::Array & train_data, py::list & randomizer) {
            Vector<candy::Randomizer> cpp_randomizer(pyseq_to_vector<candy::Randomizer>(randomizer));
            self.initialize(train_data, cpp_randomizer.data());
        },
        "Initialize values of model based on train data.",
        py::arg("train_data"), py::arg("randomizer")
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
            Lock the file when writing to prevent data race. The lock action may cause a delay.
        )",
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
            Lock the file when writing to prevent data race. The lock action may cause a delay.
        )",
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
            Loss function to calculate the gradient.
        )",
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
        "Allocate memory and make the gradient valid."
    );
    gradient_pyclass.def(
        "__exit__",
        [](candy::Gradient & self, py::object type, py::object value, py::object traceback) {
            delete[] self.value().data();
            self.value().data() = nullptr;
        },
        "De-allocate memory and make the gradient invalid."
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
            Data to calculate the gradient.
        )",
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

        This memory view can be transformed into list or Numpy array as demanded by user.
        )"
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
    // representation
    optimizer_pyclass.def(
        "__repr__",
        [](const candy::Optimizer & self) { return self.str(); }
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
            Learning rate.
        )",
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
            Bias.
        )",
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
            Bias.
        )",
        py::arg("learning_rate"), py::arg("beta_m"), py::arg("beta_v"), py::arg("model"), py::arg("bias") = 1e-8
    );
    // create adadelta
    candy_module.def(
        "create_adadelta",
        [](double learning_rate, double rho, const candy::Model & model, double bias) {
            return new candy::Optimizer(candy::create_adadelta(learning_rate, rho, model, bias));
        },
        R"(
        Create an optimizer with AdaDelta algorithm.

        See also :cpp:class:`merlin::candy::optmz::AdaDelta`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.
        rho : float
            Decay constant.
        model : merlin.candy.Model
            Model to fit.
        bias : float, default=1e-8
            Bias.
        )",
        py::arg("learning_rate"), py::arg("rho"), py::arg("model"), py::arg("bias") = 1e-8
    );
    // create rmsprop
    candy_module.def(
        "create_rmsprop",
        [](double learning_rate, double beta, const candy::Model & model, double bias) {
            return new candy::Optimizer(candy::create_rmsprop(learning_rate, beta, model, bias));
        },
        R"(
        Create an optimizer with RmsProp algorithm.

        See also :cpp:class:`merlin::candy::optmz::RmsProp`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.
        beta : float
            Decay constant.
        model : merlin.candy.Model
            Model to fit.
        bias : float, default=1e-16
            Bias to avoid division by zero.
        )",
        py::arg("learning_rate"), py::arg("beta"), py::arg("model"), py::arg("bias") = 1e-16
    );
}

// Wrap merlin::candy::TrialPolicy
void wrap_trial_policy(py::module & candy_module) {
    auto trial_policy_pyclass = py::class_<candy::TrialPolicy>(
        candy_module,
        "TrialPolicy",
        R"(
        Trial policy for test run (dry run).

        Wrapper of :cpp:class:`merlin::candy::TrialPolicy`.)"
    );
    // constructor
    trial_policy_pyclass.def(
        py::init(
            [](std::uint64_t discarded, std::uint64_t strict, std::uint64_t loose) {
                return new candy::TrialPolicy(discarded, strict, loose);
            }
        ),
        R"(
        Constructor from attributes.

        Parameters
        ----------
        discarded : int, default=1
            Number of discarded steps.
        strict : int, default=199
            Number of steps with strictly descent of error.
        loose : int, default=800
            Number of steps tolerated for random rounding error near local minima.
        )",
        py::arg("discarded") = 1, py::arg("strict") = 199, py::arg("loose") = 800
    );
    // attributes
    trial_policy_pyclass.def_property_readonly(
        "discarded",
        [](const candy::TrialPolicy & self) { return self.discarded(); },
        "Get number of discarded steps."
    );
    trial_policy_pyclass.def_property_readonly(
        "strict",
        [](const candy::TrialPolicy & self) { return self.strict(); },
        "Get number of steps with strictly descent of error."
    );
    trial_policy_pyclass.def_property_readonly(
        "loose",
        [](const candy::TrialPolicy & self) { return self.loose(); },
        "Get number of steps tolerated for random rounding error near local minima."
    );
    trial_policy_pyclass.def_property_readonly(
        "sum",
        [](const candy::TrialPolicy & self) { return self.sum(); },
        "Get total number of steps."
    );
    // representation
    trial_policy_pyclass.def(
        "__repr__",
        [](const candy::TrialPolicy & self) { return self.str(); }
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
            [](const candy::Model & model, const candy::Optimizer & optimizer, Synchronizer & synch) {
                return new candy::Trainer(model, optimizer, synch);
            }
        ),
        R"(
        Construct a trainer from model and optimizer.

        Parameters
        ----------
        model : merlin.candy.Model
            Model to train.
        optimizer : merlin.candy.Optimizer
            Optimizer training the model.
        synch : merlin.Synchronizer
            Asynchronous stream to register the training process. Destroying the synchronizer before the Trainer results
            in undefined behavior.
        )",
        py::arg("model"), py::arg("optimizer"), py::arg("synch"), py::keep_alive<1,4>()
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
        "Change optimizer.",
        py::arg("new_optmz")
    );
    // change filename
    trainer_pyclass.def(
        "set_fname",
        [](candy::Trainer & self, const std::string & new_fname) { self.set_fname(new_fname); },
        "Set filename to serialize trained model.",
        py::arg("new_fname")
    );
    // update model until convergence
    trainer_pyclass.def(
        "update_until_cpu",
        [](candy::Trainer & self, const array::Array & data, std::uint64_t rep, double threshold,
           std::uint64_t n_threads, const std::string & metric, bool export_result) {
            self.update_until(data, rep, threshold, n_threads, trainmetric_map.at(metric), export_result);
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
            Training metric for the model.
        export_result : bool, default=True
            Save trained model to a file.
        )",
        py::arg("data"), py::arg("rep"), py::arg("threshold"), py::arg("n_threads") = 1,
        py::arg("metric") = "relsquare", py::arg("export_result") = true
    );
    trainer_pyclass.def(
        "update_until_gpu",
        [](candy::Trainer & self, const array::Parcel & data, std::uint64_t rep, double threshold,
           std::uint64_t n_threads, const std::string & metric, bool export_result) {
            self.update_until(data, rep, threshold, n_threads, trainmetric_map.at(metric), export_result);
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
            Training metric for the model.
        export_result : bool, default=True
            Save trained model to a file.
        )",
        py::arg("data"), py::arg("rep"), py::arg("threshold"), py::arg("n_threads") = 32,
        py::arg("metric") = "relsquare", py::arg("export_result") = true
    );
    // update model for a given number of iterations
    trainer_pyclass.def(
        "update_for_cpu",
        [](candy::Trainer & self, const array::Array & data, std::uint64_t max_iter, std::uint64_t n_threads,
           const std::string & metric, bool export_result) {
            self.update_for(data, max_iter, n_threads, trainmetric_map.at(metric), export_result);
        },
        R"(
        Update CP model according to gradient on CPU.

        Update CP model for a certain number of iterations.

        Parameters
        ----------
        data : merlin.array.Array
            Data to train the model.
        max_iter : int
            Ma number of iterations.
        n_threads : int, default=1
            Number of parallel threads for training the model.
        metric : str, default="relsquare"
            Training metric for the model.
        export_result : bool, default=True
            Save trained model to a file.
        )",
        py::arg("data"), py::arg("max_iter"), py::arg("n_threads") = 1, py::arg("metric") = "relsquare",
        py::arg("export_result") = true
    );
    trainer_pyclass.def(
        "update_for_gpu",
        [](candy::Trainer & self, const array::Parcel & data, std::uint64_t max_iter, std::uint64_t n_threads,
           const std::string & metric, bool export_result) {
            self.update_for(data, max_iter, n_threads, trainmetric_map.at(metric), export_result);
        },
        R"(
        Update CP model according to gradient on GPU.

        Update CP model for a certain number of iterations.

        Parameters
        ----------
        data : merlin.array.Parcel
            Data to train the model.
        max_iter : int
            Ma number of iterations.
        n_threads : int, default=1
            Number of parallel threads for training the model.
        metric : str, default="relsquare"
            Training metric for the model.
        export_result : bool, default=True
            Save trained model to a file.
        )",
        py::arg("data"), py::arg("max_iter"), py::arg("n_threads") = 1, py::arg("metric") = "relsquare",
        py::arg("export_result") = true
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
        [](candy::Trainer & self, const array::Array & data, const candy::TrialPolicy & policy, std::uint64_t n_threads,
           const std::string & metric) {
            DoubleVec error(policy.sum());
            UIntVec count(1);
            self.dry_run(data, error, count[0], policy, n_threads, trainmetric_map.at(metric));
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
        policy: merlin.candy.TrialPolicy, default=merlin.candy.TrialPolicy()
            Number of steps for each stage of the dry-run.
        n_threads : int, default=1
            Number of parallel threads for calculating the gradient.
        metric : str, default="relsquare"
            Training metric for the model.
        Returns
        -------
        error : np.ndarray[dtype=np.float64]
            1-D array storing error per iteration.
        count : np.ndarray[dtype=np.float64]
            1-D array of size 1, storing the number of iterations performed before breaking the dry run.
        )",
        py::arg("data"), py::arg("policy") = candy::TrialPolicy(), py::arg("n_threads") = 1,
        py::arg("metric") = "relsquare"
    );
    trainer_pyclass.def(
        "dry_run_gpu",
        [](candy::Trainer & self, const array::Parcel & data, const candy::TrialPolicy & policy,
           std::uint64_t n_threads, const std::string & metric) {
            DoubleVec error(policy.sum());
            UIntVec count(1);
            self.dry_run(data, error, count[0], policy, n_threads, trainmetric_map.at(metric));
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
        policy: merlin.candy.TrialPolicy, default=merlin.candy.TrialPolicy()
            Number of steps for each stage of the dry-run.
        n_threads : int, default=32
            Number of parallel threads for calculating the gradient.
        metric : str, default="relsquare"
            Training metric for the model.
        Returns
        -------
        error : np.ndarray[dtype=np.float64]
            1-D array storing error per iteration.
        count : np.ndarray[dtype=np.float64]
            1-D array of size 1, storing the number of iterations performed before breaking the dry run.
        )",
        py::arg("data"), py::arg("policy") = candy::TrialPolicy(), py::arg("n_threads") = 32,
        py::arg("metric") = "relsquare");
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

// Wrap merlin::candy::Randomizer enum
static void wrap_randomizer(py::module & candy_module) {
    // add candy submodule
    py::module rand_module = candy_module.def_submodule("rand", "Randomization method for Canonical Polyadic model.");
    // add Gaussian
    auto gaussian_pyclass = py::class_<candy::rand::Gaussian>(
        rand_module,
        "Gaussian",
        R"(
        Initialization by normal distribution from data mean and variance.

        Wrapper of :cpp:class:`merlin::candy::rand::Gaussian`.
        )"
    );
    gaussian_pyclass.def(
        py::init([](void) { return new candy::rand::Gaussian(); })
    );
    gaussian_pyclass.def(
        "__repr__",
        [](const candy::rand::Gaussian & self) { return self.str(); }
    );
    // add Uniform
    auto uniform_pyclass = py::class_<candy::rand::Uniform>(
        rand_module,
        "Uniform",
        R"(
        Initialization by uniform distribution from data mean.

        Wrapper of :cpp:class:`merlin::candy::rand::Uniform`.
        )"
    );
    uniform_pyclass.def(
        py::init([](double k) { return new candy::rand::Uniform(k); }),
        "Constructor from relative scale value.",
        py::arg("k") = 0.01
    );
    uniform_pyclass.def(
        "__repr__",
        [](const candy::rand::Uniform & self) { return self.str(); }
    );
}

void wrap_candy(py::module & merlin_package) {
    // add candy submodule
    py::module candy_module = merlin_package.def_submodule("candy", "Data compression by Candecomp-Paraface method.");
    // add classes
    wrap_model(candy_module);
    wrap_gradient(candy_module);
    wrap_optimizer(candy_module);
    wrap_randomizer(candy_module);
    wrap_trial_policy(candy_module);
    wrap_trainer(candy_module);
}

}  // namespace merlin
