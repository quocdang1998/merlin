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

bool stop_criterion(double before_err, double after_err) {
    double rel_err = std::abs(before_err - after_err) / before_err;
    return (rel_err > 1e-1);
}

/*
double get_max(const merlin::candy::Model & model, const merlin::array::Array & train_data) {
    merlin::Vector<double> max_vector(::omp_get_max_threads(), 0.0);
    #pragma omp parallel for
    for (std::int64_t i_point = 0; i_point < train_data.size(); i_point++) {
        merlin::intvec index = merlin::contiguous_to_ndim_idx(i_point, train_data.shape());
        double data_point = train_data.get(index);
        double error = (data_point == 0) ? 0.0 : std::abs(model.eval(index) / data_point - 1.f);
        std::uint64_t i_thread = i_point % max_vector.size();
        max_vector[i_thread] = (error > max_vector[i_thread]) ? error : max_vector[i_thread];
    }
    double result = 0.0;
    for (std::uint64_t i_thread = 0; i_thread < max_vector.size(); i_thread++) {
        result = (result > max_vector[i_thread]) ? result : max_vector[i_thread];
    }
    return result;
}
*/
int main(void) {
    double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, 2.5};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model before trained: %s\n", model.str().c_str());

    merlin::Vector<double> gradient_data(model.num_params());
    merlin::candy::Gradient grad(gradient_data.data(), model.num_params(), merlin::candy::TrainMetric::RelativeSquare);

    merlin::candy::Optimizer opt = merlin::candy::create_grad_descent(0.1);
/*
    std::uint64_t n_thread = 5;
    merlin::intvec cache(n_thread * model.ndim());
    for (std::uint64_t i = 0; i < 100; i++) {
        #pragma omp parallel num_threads(n_thread)
        {
            grad.calc_by_cpu(model, train_data, ::omp_get_thread_num(), n_thread, cache.data());
            opt.update_cpu(model, grad, ::omp_get_thread_num(), n_thread);
        }
    }
    MESSAGE("Model gradient: %s\n", grad.str().c_str());
    MESSAGE("Model eval: %f\n", model.eval({1,1}));
*/
    merlin::candy::Trainer train(model, std::move(train_data), opt);
    train.update(10, ::stop_criterion, 5, merlin::candy::TrainMetric::RelativeSquare);
    train.synchronize();
    MESSAGE("Model eval: %f\n", train.get_model().eval({1,1}));
}
