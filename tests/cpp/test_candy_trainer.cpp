// Copyright 2023 quocdang1998
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/candy/gradient.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/candy/trainer.hpp"
#include "merlin/env.hpp"
#include "merlin/logger.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main(void) {
    // create Environment
    Environment::init_cuda(0);

    // double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, 2.5};
    double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    UIntVec data_dims = {2, 3}, data_strides = {data_dims[1] * sizeof(double), sizeof(double)};
    array::Array train_data(data, data_dims, data_strides);
    Message("Data: %s\n", train_data.str().c_str());

    candy::Model model({{1.0, 0.5, 1.6, 2.7}, {2.0, 1.0, 2.4, 1.2, 4.6, 3.5}}, 2);
    Message("Model before trained: %s\n", model.str().c_str());

    // candy::Optimizer opt = candy::create_grad_descent(0.5);
    // candy::Optimizer opt = candy::create_adagrad(0.5, model);
    // candy::Optimizer opt = candy::create_adam(0.5, 0.9, 0.99, model);
    // candy::Optimizer opt = candy::create_adadelta(1, 0.9999, model);
    candy::Optimizer opt = candy::create_rmsprop(0.5, 1e-4, model);
    {
        Message m("Optimizer:");
        m << opt.str() << "\n";
    }

    Synchronizer cpu_synch(ProcessorType::Cpu);
    candy::Trainer train("FooTrainer", model, opt, cpu_synch);

    // test dry-run
    std::uint64_t max_iter = 50;
    DoubleVec error_by_step(max_iter);
    std::uint64_t real_iter;
    train.dry_run(train_data, error_by_step, real_iter, max_iter);
    cpu_synch.synchronize();
    bool test = real_iter == max_iter;
    if (!test) {
        Fatal<std::runtime_error>("Provided optimizer not compatible with the model.\n");
    }

    // test official update
    train.update(train_data, 10, 1e-2, 3, candy::TrainMetric::RelativeSquare);
    array::Array reconstructed_data(train_data.shape());
    train.reconstruct(reconstructed_data, 4);
    cpu_synch.synchronize();
    Message("Model after trained: %s\n", train.model().str().c_str());
    Message("Reconstructed: %s\n", reconstructed_data.str().c_str());

    {
        Message m("Optimizer:");
        m << train.optmz().str() << "\n";
    }
}
