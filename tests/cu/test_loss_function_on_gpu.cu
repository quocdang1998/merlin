#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/copy.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/utils.hpp"


int main(void) {
    // initialize data
    double data[6] = {1.2, 2.3, 5.7, 4.8, 7.1, 0.0};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    // copy data to GPU
    merlin::array::Parcel data_gpu(train_data.shape());
    data_gpu.transfer_data_to_gpu(train_data);

    // initialize model
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model: %s\n", model.str().c_str());

    // calculate loss fucntion
    merlin::floatvec loss_fragments(3);
    merlin::candy::calc_loss_function_gpu(model, data_gpu, loss_fragments, merlin::cuda::Stream(), 3);
    MESSAGE("Loss fucntion GPU: %f\n", loss_fragments[0] + loss_fragments[1] + loss_fragments[2]);
    MESSAGE("Value of loss function (CPU): %f\n", merlin::candy::calc_loss_function_cpu(model, train_data));

    // calculate gradient
    merlin::floatvec gradient_gpu(model.size());
    merlin::candy::calc_gradient_vector_gpu(model, data_gpu, gradient_gpu, merlin::cuda::Stream(), 3);
    MESSAGE("Gradient vector on GPU: %s\n", gradient_gpu.str().c_str());
    merlin::Vector<double> gradient_cpu = merlin::candy::calc_gradient_vector_cpu(model, train_data);
    MESSAGE("Gradient vector on CPU: %s\n", gradient_cpu.str().c_str());
}
