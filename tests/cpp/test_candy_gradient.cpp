// Copyright 2023 quocdang1998

#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/copy.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

int main(void) {
    double data[6] = {1.2, 2.3, 5.7, 4.8, 7.1, 2.5};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model: %s\n", model.str().c_str());

    for (std::uint64_t i = 0; i < train_data.size(); i++) {
        merlin::intvec index = merlin::contiguous_to_ndim_idx(i, data_dims);
        MESSAGE("Model evaluation at (%s): %f\n", index.str().c_str(), model.eval(index));
    }

    MESSAGE("Value of loss function: %f\n", merlin::candy::calc_loss_function_cpu(model, train_data));
    merlin::Vector<double> gradient = merlin::candy::calc_gradient_vector_cpu(model, train_data);
    MESSAGE("Gradient vector: %s\n", gradient.str().c_str());
}
