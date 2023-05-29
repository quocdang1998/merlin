// Copyright 2023 quocdang1998
#include <cinttypes>
#include <cmath>

#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/copy.hpp"
#include "merlin/candy/grad_descent.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

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

int main(void) {
    double data[6] = {1.2, 2.3, 3.0, 4.8, 7.1, 2.5};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    // merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    // merlin::candy::Model model(data_dims, 2);
    merlin::candy::Model model(train_data.shape(), 2);
    model.initialize(train_data, merlin::candy::RandomInitializer::UniformDistribution, 24);
    // merlin::candy::Model model({{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 2);  // do not work !!!
    MESSAGE("Model: %s\n", model.str().c_str());

    MESSAGE("Value of loss function: %f\n", merlin::candy::calc_loss_function_cpu(model, train_data));
    merlin::Vector<double> gradient(model.size());
    merlin::candy::calc_gradient_vector_cpu(model, train_data, gradient);
    MESSAGE("Gradient vector: %s\n", gradient.str().c_str());

    std::printf("\n");
    merlin::candy::Adam grad(0.5);
    // merlin::candy::AdaGrad grad(0.2);
    // merlin::candy::GradDescent grad(0.2);
    for (std::uint64_t i = 0; i < 1000; i++) {
        merlin::candy::calc_gradient_vector_cpu(model, train_data, gradient);
        grad.update_cpu(model, gradient);
        if (i % 100 == 0) {
            MESSAGE("Model update after %" PRIu64 " steps: %s\n", i, model.str().c_str());
        }
    }
    MESSAGE("Model update after last steps: %s\n", model.str().c_str());
    MESSAGE("Gradient update after last steps: %s\n", gradient.str().c_str());
    MESSAGE("Loss function after trained: %f\n", merlin::candy::calc_loss_function_cpu(model, train_data));
    MESSAGE("Max relative error after trained: %f\n", get_max(model, train_data));

    for (std::uint64_t i = 0; i < train_data.size(); i++) {
        merlin::intvec index = merlin::contiguous_to_ndim_idx(i, data_dims);
        MESSAGE("Model evaluation at (%s): %f\n", index.str().c_str(), model.eval(index));
    }
}
