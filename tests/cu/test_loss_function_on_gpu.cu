#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/copy.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/loss.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/utils.hpp"

__global__ void calc_loss_on_gpu(merlin::candy::Model * p_model, merlin::array::Parcel * p_data) {
    extern __shared__ char share_ptr[];
    merlin::candy::Model * p_model_shared = reinterpret_cast<merlin::candy::Model *>(share_ptr);
    merlin::array::Parcel * p_data_shared = reinterpret_cast<merlin::array::Parcel *>(p_model->copy_to_shared_mem(p_model_shared, p_model_shared+1));
    std::uint64_t * cache_memory = reinterpret_cast<std::uint64_t *>(p_data->copy_to_shared_mem(p_data_shared, p_data_shared+1));
    double * temporary_storage = reinterpret_cast<double *>(cache_memory + merlin::size_of_block() * p_data_shared->ndim());
    merlin::candy::calc_loss_function_gpu(p_model_shared, p_data_shared, cache_memory, temporary_storage);
    if (merlin::flatten_thread_index() == 0) {
        CUDAOUT("Value of loss function (GPU): %f\n", temporary_storage[0]);
    }
    __syncthreads();
}

__global__ void calc_gradient_on_gpu(merlin::candy::Model * p_model, merlin::array::Parcel * p_data, double * gradient_vector) {
    extern __shared__ char share_ptr[];
    merlin::candy::Model * p_model_shared = reinterpret_cast<merlin::candy::Model *>(share_ptr);
    merlin::array::Parcel * p_data_shared = reinterpret_cast<merlin::array::Parcel *>(p_model->copy_to_shared_mem(p_model_shared, p_model_shared+1));
    std::uint64_t * cache_memory = reinterpret_cast<std::uint64_t *>(p_data->copy_to_shared_mem(p_data_shared, p_data_shared+1));
    merlin::candy::calc_gradient_vector_gpu(p_model_shared, p_data_shared, cache_memory, gradient_vector);
}

int main(void) {
    // initialize data
    double data[6] = {1.2, 2.3, 5.7, 4.8, 7.1, 2.5};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    // copy data to GPU
    merlin::array::Parcel data_gpu(train_data.shape());
    data_gpu.transfer_data_to_gpu(train_data);

    // initialize model
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model: %s\n", model.str().c_str());

    // copy Model and Parcel to GPU
    merlin::array::Parcel * parcel_ptr;
    ::cudaMalloc(&parcel_ptr, data_gpu.malloc_size() + model.malloc_size());
    merlin::candy::Model * model_ptr = reinterpret_cast<merlin::candy::Model *>(data_gpu.copy_to_gpu(parcel_ptr, parcel_ptr+1));
    model.copy_to_gpu(model_ptr, model_ptr+1);

    // calculate loss function
    MESSAGE("Value of loss function (CPU): %f\n", merlin::candy::calc_loss_function_cpu(model, train_data));
    std::uint64_t n_threads = 3;
    std::uint64_t shared_mem = data_gpu.malloc_size() + model.shared_mem_size();
    shared_mem += n_threads * data_gpu.ndim() * sizeof(std::uint64_t);
    shared_mem += n_threads * sizeof(double);
    calc_loss_on_gpu<<<1, n_threads, shared_mem>>>(model_ptr, parcel_ptr);
    ::cudaDeviceSynchronize();

    // calculate gradeint on GPU
    merlin::Vector<double> cpu_gradient = merlin::candy::calc_gradient_vector_cpu(model, train_data);
    MESSAGE("Gradient vector on CPU: %s\n", cpu_gradient.str().c_str());
    merlin::Vector<double> gpu_gradient(model.size(), 0.0);
    double * gpu_gradient_ptr;
    ::cudaMalloc(&gpu_gradient_ptr, gpu_gradient.size() * sizeof(double));
    shared_mem -= n_threads * sizeof(double);
    calc_gradient_on_gpu<<<1, n_threads, shared_mem>>>(model_ptr, parcel_ptr, gpu_gradient_ptr);
    gpu_gradient.copy_from_gpu(gpu_gradient_ptr);
    MESSAGE("Gradient vector on GPU: %s\n", gpu_gradient.str().c_str());

    // free data
    ::cudaFree(parcel_ptr);
    ::cudaFree(gpu_gradient_ptr);
}
