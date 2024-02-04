// Copyright 2024 quocdang1998
#include "merlin/regpl/polynomial.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------------------------------------------------

// Copy data to a pre-allocated memory
void * regpl::Polynomial::copy_to_gpu(regpl::Polynomial * gpu_ptr, void * coeff_data_ptr,
                                      std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    regpl::Polynomial copy_on_gpu;
    // shallow copy of coefficients, orders and term index
    double * coeff_ptr = reinterpret_cast<double *>(coeff_data_ptr);
    copy_on_gpu.coeff_.data() = coeff_ptr;
    copy_on_gpu.coeff_.size() = this->size();
    std::uint64_t * order_ptr = reinterpret_cast<std::uint64_t *>(coeff_ptr + this->size());
    copy_on_gpu.order_.data() = order_ptr;
    copy_on_gpu.order_.size() = this->ndim();
    // copy data of each vector
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(coeff_ptr, this->coeff_.data(), this->size() * sizeof(double), ::cudaMemcpyHostToDevice, stream);
    ::cudaMemcpyAsync(order_ptr, this->order_.data(), this->ndim() * sizeof(std::uint64_t), ::cudaMemcpyHostToDevice,
                      stream);
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(regpl::Polynomial), ::cudaMemcpyHostToDevice, stream);
    // nullify pointer of temporary object to avoid de-allocate GPU pointer
    copy_on_gpu.coeff_.data() = nullptr;
    copy_on_gpu.order_.data() = nullptr;
    return reinterpret_cast<void *>(order_ptr + this->ndim());
}

// Copy data from GPU to CPU
void * regpl::Polynomial::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(this->coeff_.data(), data_from_gpu, this->size() * sizeof(double), ::cudaMemcpyDeviceToHost,
                      stream);
    return reinterpret_cast<void *>(data_from_gpu + this->size());
}

}  // namespace merlin
