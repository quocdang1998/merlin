// Copyright 2023 quocdang1998
#include <cinttypes>
#include <cmath>
#include <omp.h>
#include <iostream>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/candy/gradient.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/candy/trainer.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

int main(void) {
    using namespace merlin;

    // double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, 2.5};
    double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    UIntVec data_dims = {2, 3}, data_strides = {data_dims[1] * sizeof(double), sizeof(double)};
    array::Array train_data(data, data_dims, data_strides);
    Message("Data: %s\n", train_data.str().c_str());

    candy::Model model({{1.0, 0.5, 1.6, 2.7}, {2.0, 1.0, 2.4, 1.2, 4.6, 3.5}}, 2);
    Message("Model before trained: %s\n", model.str().c_str());

    candy::Optimizer opt = candy::create_grad_descent(0.1);
    // candy::Optimizer opt = candy::create_adam(0.3, 0.9, 0.99, model);
    // candy::Optimizer opt = candy::create_adadelta(20.0, 0.9, model);

    candy::Trainer train(model, opt);

    // test dry-run
    /*bool test;
    train.dry_run(100, &test, 4, candy::TrainMetric::RelativeSquare);
    train.synchronize();
    if (!test) {
        FAILURE(std::runtime_error, "Provided optimizer not compatible with the model.\n");
    }*/

    // test official update
    train.update_cpu(train_data, 10000, 1e-2, 3, candy::TrainMetric::RelativeSquare);
    train.synchronize();
    Message("Model eval: %f\n", train.model().eval({1,1}));
}
