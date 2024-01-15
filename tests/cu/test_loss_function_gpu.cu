#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"
#include "merlin/utils.hpp"

__global__ void calc_loss_gpu(merlin::candy::Model * p_model, merlin::array::Parcel * p_data, double * p_result_gpu) {
    extern __shared__ char shared_mem[];
    __shared__ double result;
    auto [buffer, p_model_shr, p_data_shr] = merlin::cuda::copy_objects(shared_mem, *p_model, *p_data);
    std::uint64_t thread_idx = merlin::flatten_thread_index(), block_size = merlin::size_of_block();
    merlin::candy::rmse_gpu(p_model_shr, p_data_shr, reinterpret_cast<std::uint64_t *>(buffer), &result,
                            thread_idx, block_size);
    if (thread_idx == 0) {
        std::printf("Result on GPU: %f\n", result);
        *p_result_gpu = result;
    }
    __syncthreads();
}

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
    double result_gpu = 1.2;
    merlin::cuda::Memory mem(0, model, train_data_gpu, result_gpu);
    merlin::candy::Model * p_model = mem.get<0>();
    merlin::array::Parcel * p_train_data = mem.get<1>();
    double * p_result_gpu = mem.get<2>();

    // calculate loss function (GPU)
    std::uint64_t num_threads_gpu = 32;
    std::uint64_t share_mem = model.sharedmem_size() + train_data_gpu.sharedmem_size();
    share_mem += model.ndim() * num_threads_gpu;
    calc_loss_gpu<<<1, num_threads_gpu, share_mem, 0>>>(p_model, p_train_data, p_result_gpu);
    merlin::cuda_mem_cpy_device_to_host(&result_gpu, p_result_gpu, sizeof(double), 0);
    merlin::cuda::Device::synchronize();
    MESSAGE("Value of loss function (GPU): %f\n", result_gpu);

    // calculate loss function (CPU)
    std::uint64_t num_threads_cpu = 2;
    merlin::intvec buffer_cpu(model.ndim() * num_threads_cpu);
    double result_cpu = merlin::candy::rmse_cpu(&model, &train_data, buffer_cpu.data(), num_threads_cpu);
    MESSAGE("Value of loss function (CPU): %f\n", result_cpu);
}
