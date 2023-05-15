// Copyright 2023 quocdang1998
#include "merlin/candy/loss.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/candy/model.hpp"  // merlin::candy::Model
#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Loss function
// --------------------------------------------------------------------------------------------------------------------

// Calculate loss function with GPU parallelism
void candy::calc_loss_function_gpu(const candy::Model & model, const array::Parcel & train_data, floatvec & result,
                                   const cuda::Stream & stream, std::uint64_t n_thread) {
    // check stream validity
    stream.check_cuda_context();
    // check size of vector
    if (result.size() != n_thread) {
        FAILURE(std::invalid_argument, "Result vector must have the size of the number of threads.\n");
    }
    // copy data to GPU
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    cuda::Memory mem(stream.get_stream_ptr(), model, train_data);
    candy::Model * ptr_model_on_gpu = mem.get<0>();
    array::Parcel * ptr_train_data_on_gpu = mem.get<1>();
    double * ptr_result_on_gpu;
    ::cudaMallocAsync(&ptr_result_on_gpu, n_thread * sizeof(double), cuda_stream);
    std::uint64_t shared_mem_size = model.shared_mem_size() + train_data.malloc_size();
    shared_mem_size += n_thread * (model.ndim() * sizeof(std::uint64_t) + sizeof(double));
    // call CUDA kernel
    candy::call_loss_function_kernel(ptr_model_on_gpu, ptr_train_data_on_gpu, ptr_result_on_gpu,
                                     shared_mem_size, stream.get_stream_ptr(), n_thread);
    // copy the result back to CPU
    result.copy_from_gpu(ptr_result_on_gpu, stream.get_stream_ptr());
    ::cudaFreeAsync(ptr_result_on_gpu, cuda_stream);
}

// --------------------------------------------------------------------------------------------------------------------
// Model gradient
// --------------------------------------------------------------------------------------------------------------------

// Calculate gradient of canonical decomposition model with CPU parallelism
void candy::calc_gradient_vector_gpu(const candy::Model & model, const array::Parcel & train_data, floatvec & result,
                                    const cuda::Stream & stream, std::uint64_t n_thread) {
    // check stream validity
    stream.check_cuda_context();
    // check size of vector
    if (result.size() != model.size()) {
        FAILURE(std::invalid_argument, "Result vector must have the size of the number of parameters in the model.\n");
    }
    // copy data to GPU
    ::cudaStream_t cuda_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    cuda::Memory mem(stream.get_stream_ptr(), model, train_data);
    candy::Model * ptr_model_on_gpu = mem.get<0>();
    array::Parcel * ptr_train_data_on_gpu = mem.get<1>();
    double * ptr_gradient_on_gpu;
    ::cudaMallocAsync(&ptr_gradient_on_gpu, result.size() * sizeof(double), cuda_stream);
    std::uint64_t shared_mem_size = model.shared_mem_size() + train_data.malloc_size();
    shared_mem_size += n_thread * (model.ndim() * sizeof(std::uint64_t) + sizeof(double));
    // call CUDA kernel
    candy::call_model_gradient_kernel(ptr_model_on_gpu, ptr_train_data_on_gpu, ptr_gradient_on_gpu,
                                      shared_mem_size, stream.get_stream_ptr(), n_thread);
    // deallocate memeory and return result
    result.copy_from_gpu(ptr_gradient_on_gpu, stream.get_stream_ptr());
    ::cudaFreeAsync(ptr_gradient_on_gpu, cuda_stream);
}

}  // namespace merlin
