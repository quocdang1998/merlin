// Copyright 2023 quocdang1998
#include "py_api.hpp"

#include <cstddef>  // nullptr
#include <cstring>  // std::memset

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
        "shape",
        [](const candy::Model & self) { return array_to_pylist(self.shape()); },
        "Get shape."
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
            using RandomizerVector = vector::StaticVector<candy::Randomizer, max_dim>;
            RandomizerVector cpp_randomizer(pyseq_to_array<candy::Randomizer>(randomizer));
            self.initialize(train_data, cpp_randomizer.data());
        },
        "Initialize values of model based on train data.", py::arg("train_data"), py::arg("randomizer"));
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
            self.value().assign(gradient_memory, self.value().size());
            return &self;
        },
        "Allocate memory and make the gradient valid."
    );
    gradient_pyclass.def(
        "__exit__",
        [](candy::Gradient & self, py::object type, py::object value, py::object traceback) {
            delete[] self.value().data();
        },
        "De-allocate memory and make the gradient invalid."
    );
    // calculate gradient
    gradient_pyclass.def(
        "calc",
        [](candy::Gradient & self, candy::Model & model, const array::Array & data) {
            Index index;
            self.calc_by_cpu(model, data, index);
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
    // optimizer
    auto optimizer_pyclass = py::class_<candy::Optimizer>(
        candy_module,
        "Optimizer",
        R"(
        Algorithm for updating a model based on its gradient.

        Wrapper of :cpp:class:`merlin::candy::Optimizer`.)"
    );
    optimizer_pyclass.def(
        py::init(
            [](void) {
                return new candy::Optimizer();
            }
        ),
        R"(
        Default constructor.)"
    );
    optimizer_pyclass.def(
        "__repr__",
        [](const candy::Optimizer & self) { return self.str(); }
    );
    // add optmz submodule
    py::module optmz_module = candy_module.def_submodule("optmz", "Optimization methods for training CP models.");
    // create grad descent
    optmz_module.def(
        "create_grad_descent",
        [](double learning_rate) {
            return new candy::Optimizer(candy::optmz::create_grad_descent(learning_rate));
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
    optmz_module.def(
        "create_adagrad",
        [](double learning_rate, std::uint64_t num_params, double bias) {
            return new candy::Optimizer(candy::optmz::create_adagrad(learning_rate, num_params, bias));
        },
        R"(
        Create an optimizer with AdaGrad algorithm.

        See also :cpp:class:`merlin::candy::optmz::AdaGrad`.

        Parameters
        ----------
        learning_rate : float
            Learning rate.
        num_params : int
            Number of parameters of the model.
        bias : float, default=1e-8
            Bias.
        )",
        py::arg("learning_rate"), py::arg("model"), py::arg("bias") = 1e-8
    );
    // create adam
    optmz_module.def(
        "create_adam",
        [](double learning_rate, double beta_m, double beta_v, std::uint64_t num_params, double bias) {
            return new candy::Optimizer(candy::optmz::create_adam(learning_rate, beta_m, beta_v, num_params, bias));
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
        num_params : int
            Number of parameters of the model.
        bias : float, default=1e-8
            Bias.
        )",
        py::arg("learning_rate"), py::arg("beta_m"), py::arg("beta_v"), py::arg("model"), py::arg("bias") = 1e-8
    );
    // create adadelta
    optmz_module.def(
        "create_adadelta",
        [](double learning_rate, double rho, std::uint64_t num_params, double bias) {
            return new candy::Optimizer(candy::optmz::create_adadelta(learning_rate, rho, num_params, bias));
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
        num_params : int
            Number of parameters of the model.
        bias : float, default=1e-8
            Bias.
        )",
        py::arg("learning_rate"), py::arg("rho"), py::arg("num_params"), py::arg("bias") = 1e-8
    );
    // create rmsprop
    optmz_module.def(
        "create_rmsprop",
        [](double learning_rate, double beta, std::uint64_t num_params, double bias) {
            return new candy::Optimizer(candy::optmz::create_rmsprop(learning_rate, beta, num_params, bias));
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
        num_params : int
            Number of parameters of the model.
        bias : float, default=1e-16
            Bias to avoid division by zero.
        )",
        py::arg("learning_rate"), py::arg("beta"), py::arg("num_params"), py::arg("bias") = 1e-16
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
        Launch a train process on CP model asynchronously.

        Wrapper of :cpp:class:`merlin::candy::Trainer`.)"
    );
    // constructor
    trainer_pyclass.def(
        py::init(
            [](std::uint64_t capacity, Synchronizer & synch) {
                return new candy::Trainer(capacity, synch);
            }
        ),
        R"(
        Construct a trainer from the total number of elements.

        Parameters
        ----------
        capacity : int
            Number of maximum cases to train.
        synch : merlin.Synchronizer
            Asynchronous stream to register the training process. Destroying the synchronizer before the Trainer results
            in undefined behavior.
        )",
        py::arg("capacity"), py::arg("synch"), py::keep_alive<1,3>()
    );
    // add elements
    trainer_pyclass.def_property_readonly(
        "is_full",
        [](candy::Trainer & self) { return self.is_full(); },
        "Check if the trainer is filled."
    );
    trainer_pyclass.def(
        "set_model",
        [](candy::Trainer & self, const std::string & name, const candy::Model & model) {
            self.set_model(name, model);
        },
        "Add or modify the model assigned to an ID.",
        py::arg("name"), py::arg("model")
    );
    trainer_pyclass.def(
        "set_optmz",
        [](candy::Trainer & self, const std::string & name, const candy::Optimizer & optmz) {
            self.set_optmz(name, optmz);
        },
        "Add or modify the optimizer assigned to an ID.",
        py::arg("name"), py::arg("optmz")
    );
    trainer_pyclass.def(
        "set_data",
        [](candy::Trainer & self, const std::string & name, const array::NdData & data) {
            self.set_data(name, data);
        },
        "Add or modify the optimizer assigned to an ID.",
        py::arg("name"), py::arg("data")
    );
    trainer_pyclass.def(
        "set_export_fname",
        [](candy::Trainer & self, const std::string & name, const std::string & export_fname) {
            self.set_export_fname(name, export_fname);
        },
        "Add or modify the output filename of a model.",
        py::arg("name"), py::arg("export_fname")
    );
    // get elements
    trainer_pyclass.def(
        "get_keys",
        [](candy::Trainer & self) {
            std::vector<std::string> keys = self.get_keys();
            py::list key_list;
            for (std::string & key : keys) {
                key_list.append(key);
            }
            return key_list;
        },
        "Get list of keys."
    );
    trainer_pyclass.def_property_readonly(
        "on_gpu",
        [](candy::Trainer & self) { return self.on_gpu(); },
        "Query if the object data is instatialized on CPU or on GPU."
    );
    trainer_pyclass.def(
        "get_model",
        [](candy::Trainer & self, const std::string & name) { return self.get_model(name); },
        R"(
        Get copy to a model.

        This function is synchronized with the calling CPU thread. Only invoke it after all training processes are
        finished to avoid data race.
        )",
        py::arg("name")
    );
    // training
    trainer_pyclass.def(
        "dry_run",
        [](candy::Trainer & self, py::list & names, const candy::TrialPolicy & policy, std::uint64_t n_threads,
           const std::string & metric) {
            // create tracking map
            py::dict py_map;
            std::map<std::string, std::pair<double *, std::uint64_t *>> tracking_map;
            std::uint64_t error_size = policy.sum();
            for (auto it = names.begin(); it != names.end(); ++it) {
                py::handle element = *it;
                std::string name = element.cast<std::string>();
                double * error = new double[error_size];
                std::memset(error, 0, error_size * sizeof(double));
                std::uint64_t * count = new std::uint64_t[1];
                *count = 0;
                py_map[name.c_str()] = py::make_tuple(make_wrapper_array<double>(error, error_size),
                                                      make_wrapper_array<std::uint64_t>(count, 1));
                tracking_map[name] = {error, count};
            }
            // launch
            self.dry_run(tracking_map, policy, n_threads, trainmetric_map.at(metric));
            return py_map;
        },
        R"(
        Dry-run.

        Perform the gradient descent algorithm for a given number of iterations without updating the model. The
        iterative process will automatically stop when the RMSE after the update is larger than before.

        Parameters
        ----------
        names : List[str]
            List of keys of cases to launch the dry-run.
        policy : merlin.candy.TrialPolicy, default=merlin.candy.TrialPolicy()
            Number of steps for each stage of the dry-run.
        n_threads : int, default=16
            Number of OpenMP threads, or number of CUDA threads per block.
        metric : str, default="relsquare"
            Training metric for the model.

        Returns
        -------
        tracking_map: ``Dict[str, Tuple[numpy.ndarray, numpy.ndarray]]``
            Dictionary of provided keys to error array and count. Note that the result is only ready after
            synchronization.
        )",
        py::arg("names"), py::arg("policy") = candy::TrialPolicy(), py::arg("n_threads") = 16,
        py::arg("metric") = "relsquare"
    );
    trainer_pyclass.def(
        "update_until",
        [](candy::Trainer & self, std::uint64_t rep, double threshold, std::uint64_t n_threads,
           const std::string & metric, bool export_result) {
            self.update_until(rep, threshold, n_threads, trainmetric_map.at(metric), export_result);
        },
        R"(
        Update the CP model according to the gradient until a specified threshold is met.

        Update CP model for a certain number of iterations, and check if the relative error after the training
        process is smaller than a given threshold. If this is the case, break the training. Otherwise, continue to train
        again and check.

        This function is asynchronous. To get the model after trained, remember to synchronize the object first.

        Parameters
        ----------
        rep : int
            Number of times to repeat the gradient descent update in each step.
        threshold : float
            Threshold to stop the training process.
        n_threads : int, default=16
            Number of OpenMP threads, or number of CUDA threads per block.
        metric : str, default="relsquare"
            Training metric for the model.
        export_result : bool, default=False
            Flag indicate whether to serialize the model right at the end of the training (must be ``False`` in GPU
            configuration).
        )",
        py::arg("rep"), py::arg("threshold"), py::arg("n_threads") = 16, py::arg("metric") = "relsquare",
        py::arg("export_result") = false
    );
    trainer_pyclass.def(
        "update_for",
        [](candy::Trainer & self, std::uint64_t max_iter, std::uint64_t n_threads,
           const std::string & metric, bool export_result) {
            self.update_for(max_iter, n_threads, trainmetric_map.at(metric), export_result);
        },
        R"(
        Update CP model according to gradient for a given number of iterations.

        Update CP model for a certain number of iterations.

        This function is asynchronous. To get the model after trained, remember to synchronize the object first.

        Parameters
        ----------
        max_iter : int
            Max number of iterations.
        n_threads : int, default=16
            Number of OpenMP threads, or number of CUDA threads per block.
        metric : str, default="relsquare"
            Training metric for the model.
        export_result : bool, default=False
            Flag indicate whether to serialize the model right at the end of the training (must be ``False`` in GPU
            configuration).
        )",
        py::arg("max_iter"), py::arg("n_threads") = 16, py::arg("metric") = "relsquare",
        py::arg("export_result") = false
    );
    // reconstruct
    trainer_pyclass.def(
        "reconstruct",
        [](candy::Trainer & self, std::uint64_t n_threads) {
            // create reconstruction map
            py::dict py_map;
            std::map<std::string, array::NdData *> rec_data_map;
            bool on_gpu = self.on_gpu();
            std::vector<std::string> keys = self.get_keys();
            for (const std::string & name : keys) {
                const auto & [shape, rank] = self.get_model_shape(name);
                array::NdData * rec_data;
                if (!on_gpu) {
                    rec_data = new array::Array(shape);
                } else {
                    rec_data = new array::Parcel(shape, std::get<cuda::Stream>(self.get_synch().core));
                }
                py_map[name.c_str()] = rec_data;
                rec_data_map[name] = rec_data;
            }
            // launch
            self.reconstruct(rec_data_map, n_threads);
            return py_map;
        },
        R"(
        Reconstruct a whole multi-dimensional data from the model using CPU parallelism.

        Parameters
        ----------
        n_threads : int, default=16
            Number of OpenMP threads, or number of CUDA threads per block.

        Returns
        -------
        rec_data_map: ``Dict[str, Union[merlin.array.Array, merlin.array.Parcel]]``
            Dictionary to reconstructed array. The type of the array is determined based on where the trainer is
            initialized.
        )",
        py::arg("n_threads") = 16
    );
    trainer_pyclass.def(
        "get_error",
        [](candy::Trainer & self, std::uint64_t n_threads) {
            // create error map
            py::dict py_map;
            std::map<std::string, std::array<double *, 2>> error_map;
            std::vector<std::string> keys = self.get_keys();
            for (const std::string & name : keys) {
                double * errors = new double[2];
                py_map[name.c_str()] = make_wrapper_array<double>(errors, 2);
                error_map[name] = {errors, errors + 1};
            }
            // launch
            self.get_error(error_map, n_threads);
            return py_map;
        },
        R"(
        Get the RMSE and RMAE error with respect to the training data.

        Parameters
        ----------
        n_threads : int, default=16
            Number of OpenMP threads, or number of CUDA threads per block.

        Returns
        -------
        error_map: ``Dict[str, numpy.ndarray]``
            Dictionary to a size-2 array storing RMSE and RMAE.
        )",
        py::arg("n_threads") = 16
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
    py::module candy_module = merlin_package.def_submodule("candy", "Data compression by CANDECOMP-PARAFAC method.");
    // add classes
    wrap_model(candy_module);
    wrap_gradient(candy_module);
    wrap_optimizer(candy_module);
    wrap_randomizer(candy_module);
    wrap_trial_policy(candy_module);
    wrap_trainer(candy_module);
}

}  // namespace merlin
