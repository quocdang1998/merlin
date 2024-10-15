// Copyright 2023 quocdang1998
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <map>
#include <chrono>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/candy/randomizer.hpp"
#include "merlin/candy/gradient.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/candy/train/cpu_trainer.hpp"
#include "merlin/candy/trainer.hpp"
#include "merlin/candy/trial_policy.hpp"
#include "merlin/logger.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main(void) {
    // double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, 2.5};
    double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    Index data_dims = {2, 3}, data_strides = {data_dims[1] * sizeof(double), sizeof(double)};
    array::Array train_data(data, data_dims, data_strides);
    Message("Data: ") << train_data.str() << "\n";

    candy::Model model({{1.0, 0.5, 1.6, 2.7}, {2.0, 1.0, 2.4, 1.2, 4.6, 3.5}}, 2);
    std::array<candy::Randomizer, 2> randomizer = {candy::rand::Gaussian(), candy::rand::Gaussian()};
    model.initialize(train_data, randomizer.data());
    Message("Model before trained: {}\n", model.str());

    // candy::Optimizer opt = candy::optmz::create_grad_descent(0.5);
    candy::Optimizer opt = candy::optmz::create_adagrad(0.3, model.num_params());
    // candy::Optimizer opt = candy::optmz::create_adam(0.005, 0.99, 0.9999, model.num_params());
    // candy::Optimizer opt = candy::optmz::create_adadelta(3, 0.9999, model.num_params());
    // candy::Optimizer opt = candy::optmz::create_rmsprop(0.02, 0.99, model.num_params());
    {
        Message m("Optimizer:");
        m << opt.str() << "\n";
    }

    Synchronizer cpu_synch(ProcessorType::Cpu);
    candy::train::CpuTrainer cpu_train(4, cpu_synch);
    cpu_train.set_model("foo", model);
    cpu_train.set_optmz("foo", opt);
    cpu_train.set_data("foo", train_data);
    candy::TrialPolicy policy(2, 8, 40);
    std::uint64_t valid_result;
    DoubleVec error(policy.sum());
    std::map<std::string, std::pair<double *, std::uint64_t *>> dryrun_map;
    dryrun_map["foo"] = {error.data(), &valid_result};
    cpu_train.dry_run(dryrun_map, policy);
    cpu_synch.synchronize();
    if (valid_result != policy.sum()) {
        Warning("Failed dryrun with: ") << error.str() << "\n";
        Fatal<std::runtime_error>("Invalid optimizer.\n");
    }
    // cpu_train.set_export_fname("foo", "foo.txt");
    // cpu_train.update_until(1000, 0.01, 3, candy::TrainMetric::RelativeSquare, true);
    cpu_train.update_for(20000, 3, candy::TrainMetric::RelativeSquare, false);
    cpu_synch.synchronize();
    std::printf("Model new: {}\n", cpu_train.get_model("foo").str());
    array::Array destination(train_data.shape());
    std::map<std::string, array::Array *> destination_map;
    destination_map["foo"] = &destination;
    cpu_train.reconstruct(destination_map, 3);
    double rmse, rmae;
    std::map<std::string, std::array<double *, 2>> error_map;
    error_map["foo"] = {&rmse, &rmae};
    cpu_train.get_error(error_map, 3);
    cpu_synch.synchronize();
    std::printf("Reconstructed data: {}\n", destination.str());
    std::printf("Reconstructed error: {} {}\n", rmse, rmae);

    // save and load model
    candy::Model & trained_model = cpu_train.get_model("foo");
    Message("Model to save: {}\n", trained_model.str());
    trained_model.save("mdl.txt", 0, true);
    candy::Model loaded_model;
    loaded_model.load("mdl.txt", 0, true);
    Message("Model to load: {}\n", loaded_model.str());
}
