#include <cstdio>
#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/candy/gradient.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/candy/optmz/grad_descent.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
/*
__global__ void call_optimizer_updater(merlin::candy::Model * p_model, merlin::candy::Optimizer * p_opt,
                                       double * p_gradient, std::uint64_t size) {
    p_opt->update_gpu(p_model, p_gradient, size);
}
*/

__global__ void test_on_gpu(merlin::candy::Model * p_model, merlin::array::Parcel * p_train_data, merlin::candy::Optimizer * p_opt) {
    extern __shared__ char share_mem[];
    auto [end_ptr, p_model_shr, p_data_shr, p_opt_shr] = merlin::cuda::copy_class_to_shared_mem(share_mem, *p_model, *p_train_data, *p_opt);
    merlin::candy::Gradient gradient(reinterpret_cast<double *>(end_ptr), p_model_shr->num_params(),
                                     merlin::candy::TrainMetric::RelativeSquare);
    merlin::intvec cache_mem;
    cache_mem.assign(reinterpret_cast<std::uint64_t *>(gradient.value().end()), 3 * p_model_shr->ndim());
    std::uint64_t thread_idx = merlin::flatten_thread_index(), block_size = merlin::size_of_block();
    gradient.calc_by_gpu(*p_model_shr, *p_data_shr, thread_idx, block_size, cache_mem.data());
    p_opt_shr->update_gpu(*p_model_shr, gradient, thread_idx, block_size);
    // copy back to global mem
    p_model_shr->copy_by_block(p_model, p_model+1, thread_idx, block_size);
}

int main(void) {
    // Initialize object on CPU
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    merlin::candy::Optimizer optimizer = merlin::candy::create_grad_descent(0.1);
    

    // Calculate gradient vector by CPU
    double data[6] = {1.2, 2.3, 5.7, 4.8, 7.1, 0.0};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    merlin::candy::Model model_cpu(model);
    merlin::floatvec cache_cpu(model_cpu.num_params());
    merlin::candy::Gradient gradient(cache_cpu.data(), model_cpu.num_params(), merlin::candy::TrainMetric::RelativeSquare);
    merlin::intvec cache_idx_cpu(model_cpu.ndim());
    gradient.calc_by_cpu(model_cpu, train_data, 0, 1, cache_idx_cpu.data());
    optimizer.update_cpu(model_cpu, gradient, 0, 1);
    MESSAGE("Model after update CPU: %s\n", model_cpu.str().c_str());

    // copy data to GPU
    merlin::array::Parcel train_data_gpu(train_data.shape());
    train_data_gpu.transfer_data_to_gpu(train_data);
    merlin::cuda::Memory mem(0, model, train_data_gpu, optimizer);
    std::uint64_t shared_mem_size = sizeof(double) * model.num_params() + sizeof(std::uint64_t) * 3 * model.ndim();
    shared_mem_size += model.sharedmem_size() + train_data_gpu.sharedmem_size() + optimizer.sharedmem_size();
    test_on_gpu<<<1, 1, shared_mem_size, 0>>>(mem.get<0>(), mem.get<1>(), mem.get<2>());
    merlin::candy::Model model_gpu(model);
    model_gpu.copy_from_gpu(reinterpret_cast<double *>(mem.get<0>()+1));
    // ::cudaDeviceSynchronize();
    MESSAGE("Model after update GPU: %s\n", model_gpu.str().c_str());
/*
    // Copy gradient vector to GPU
    double * gradient_gpu;
    cudaMalloc(&gradient_gpu, sizeof(double) * gradient.size());
    cudaMemcpy(gradient_gpu, gradient.data(), sizeof(double) * gradient.size(), cudaMemcpyHostToDevice);

    // Call updater
    call_optimizer_updater<<<1, 3>>>(gpu_model, gpu_opt, gradient_gpu, gradient.size());
    cudaDeviceSynchronize();

    // free memory
    cudaFree(gradient_gpu);
    merlin::candy::GradDescent::delete_object_on_gpu(gpu_opt);
    cudaFree(gpu_model);
*/
}
