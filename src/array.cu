// Copyright 2022 quocdang1998
#include "merlin/array.hpp"

namespace merlin {

// ------------------------------------------------------------------------------------------------
// Array (GPU)
// ------------------------------------------------------------------------------------------------

void Array::sync_to_gpu(cudaStream_t stream) {
    // allocate data on GPU
    if (this->gpu_data_ == NULL) {
        cudaMalloc(&(this->gpu_data_), sizeof(float) * this->size());
    }

    // GPU stride array
    std::vector<unsigned int> gpu_strides_(this->ndim_, 0);
    gpu_strides_[this->ndim_ - 1] = sizeof(float);
    for (int i = this->ndim_ - 2; i >= 0; i--) {
        gpu_strides_[i] = gpu_strides_[i + 1] * this->dims_[i + 1];
    }

    // copy data to GPU
    if (this->break_index_ == -1) {  // original array is perfectly contiguous
        cudaMemcpy(this->gpu_data_, this->data_,
            this->longest_contiguous_segment_, cudaMemcpyHostToDevice);
    } else {  // copy each longest_contiguous_segment
        for (Array::iterator it = this->begin(); it != this->end();) {
            // cpu leap
            unsigned int cpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                cpu_leap += it.index()[i] * this->strides_[i];
            }

            // gpu leap
            unsigned int gpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                gpu_leap += it.index()[i] * gpu_strides_[i];
            }

            // clone data
            uintptr_t src_ptr = (uintptr_t) this->data_ + cpu_leap;
            uintptr_t des_ptr = (uintptr_t) this->gpu_data_ + gpu_leap;
            cudaMemcpy(reinterpret_cast<float*>(des_ptr), reinterpret_cast<float*>(src_ptr),
                this->longest_contiguous_segment_, cudaMemcpyHostToDevice);

            // increment iterator
            it.index()[this->break_index_] += 1;
            try {
                it.update();
            }
            catch (const std::out_of_range& err) {
                break;
            }
        }
    }
}

void Array::sync_from_gpu(cudaStream_t stream) {
    // GPU stride array
    std::vector<unsigned int> gpu_strides_(this->ndim_, 0);
    gpu_strides_[this->ndim_ - 1] = sizeof(float);
    for (int i = this->ndim_ - 2; i >= 0; i--) {
        gpu_strides_[i] = gpu_strides_[i + 1] * this->dims_[i + 1];
    }

    // copy data from GPU
    if (this->break_index_ == -1) {  // original array is perfectly contiguous
        cudaMemcpy(this->data_, this->gpu_data_,
            this->longest_contiguous_segment_, cudaMemcpyDeviceToHost);
    } else {  // copy each longest_contiguous_segment
        for (Array::iterator it = this->begin(); it != this->end();) {
            // cpu leap
            unsigned int cpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                cpu_leap += it.index()[i] * this->strides_[i];
            }

            // gpu leap
            unsigned int gpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                gpu_leap += it.index()[i] * gpu_strides_[i];
            }

            // clone data
            uintptr_t src_ptr = (uintptr_t) this->gpu_data_ + gpu_leap;
            uintptr_t des_ptr = (uintptr_t) this->data_ + cpu_leap;
            cudaMemcpy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr),
                this->longest_contiguous_segment_, cudaMemcpyDeviceToHost);

            // increment iterator
            it.index()[this->break_index_] += 1;
            try {
                it.update();
            }
            catch (const std::out_of_range& err) {
                break;
            }
        }
    }
}

}  // namespace merlin
