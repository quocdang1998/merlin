#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"


int main(void) {
    // preapre data
    double data[6] = {1.2, 2.3, 3.0, 4.8, 7.1, 2.5};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    // prepare model
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model: %s\n", model.str().c_str());

    // copy data and model to GPU
    merlin::array::Parcel train_data_gpu(train_data.shape());
    train_data_gpu.transfer_data_to_gpu(train_data);
    merlin::cuda::Memory mem(0, model, train_data_gpu);
    merlin::candy::Model * p_model = mem.get<0>();
    merlin::array::Parcel * p_train_data = mem.get<1>();

    // calculate loss function
    std::uint64_t share_mem = model.sharedmem_size() + train_data_gpu.sharedmem_size();
    MESSAGE("Value of loss function (GPU): %f\n", merlin::candy::rmse_gpu(p_model, p_train_data, 2, share_mem, 32));
    MESSAGE("Value of loss function (CPU): %f\n", merlin::candy::rmse_cpu(&model, &train_data, 8));
}
