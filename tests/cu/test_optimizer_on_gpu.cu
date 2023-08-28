#include <cstdio>
#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/candy/grad_descent.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/logger.hpp"

__global__ void call_optimizer_updater(merlin::candy::Model * p_model, merlin::candy::Optimizer * p_opt,
                                       double * p_gradient, std::uint64_t size) {
    p_opt->update_gpu(p_model, p_gradient, size);
}

int main(void) {
    // Initialize model on GPU
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    merlin::candy::Model * gpu_model;
    cudaMalloc(&gpu_model, model.malloc_size());
    model.copy_to_gpu(gpu_model, gpu_model+1);

    // Initialize optimizer on GPU
    merlin::candy::GradDescent * gpu_opt = merlin::candy::GradDescent::create_object_on_gpu(0.2);

    // Calculate gradient vector
    double data[6] = {1.2, 2.3, 5.7, 4.8, 7.1, 0.0};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    merlin::Vector<double> gradient(model.size());
    merlin::candy::calc_gradient_vector_cpu(model, train_data, gradient);
    MESSAGE("Gradient vector: %s.\n", gradient.str().c_str());

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
}
