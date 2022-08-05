// Copyright 2022 quocdang1998
#include "merlin/array.hpp"

#include <cstdint>

#include "merlin/logger.hpp"

namespace merlin {

// ------------------------------------------------------------------------------------------------
// Array (GPU)
// ------------------------------------------------------------------------------------------------

void Array::sync_to_gpu(float * gpu_pdata, cudaStream_t stream) {
    // allocate data on GPU
    if (gpu_pdata == NULL) {
        this->gpu_data_.push_back(NULL);
        cudaError_t err_ = cudaMalloc(&(this->gpu_data_.back()), sizeof(float) * this->size());
        if (err_ != cudaSuccess) {
            FAILURE("Memory allocation failed with message \"%s\".", cudaGetErrorString(err_));
        }
        gpu_pdata = this->gpu_data_.back();
    }

    // GPU stride array
    std::vector<unsigned int> gpu_strides_ = contiguous_strides(this->dims_, sizeof(float));

    // longest cntiguous segment and break index
    unsigned int longest_contiguous_segment_;
    int break_index_;
    std::tie(longest_contiguous_segment_, break_index_) = lcseg_and_brindex(this->dims_,
                                                                            this->strides_);

    // copy data to GPU
    if (break_index_ == -1) {  // original array is perfectly contiguous
        cudaMemcpy(gpu_pdata, this->data_,
                   longest_contiguous_segment_, cudaMemcpyHostToDevice);
    } else {  // copy each longest_contiguous_segment
        unsigned int cpu_leap = 0;
        unsigned int gpu_leap = 0;
        for (Array::iterator it = this->begin(); it != this->end();) {
            cpu_leap = leap(it.index(), this->strides_);
            uintptr_t src_ptr = (uintptr_t) this->data_ + cpu_leap;
            gpu_leap = leap(it.index(), gpu_strides_);
            uintptr_t des_ptr = (uintptr_t) gpu_pdata + gpu_leap;
            cudaError_t err_ = cudaMemcpyAsync(reinterpret_cast<float *>(des_ptr),
                                               reinterpret_cast<float *>(src_ptr),
                                               longest_contiguous_segment_,
                                               cudaMemcpyHostToDevice);
            if (err_ != cudaSuccess) {
                FAILURE("Memory copy failed with message \"%s\".", cudaGetErrorString(err_));
            }
            it.index()[break_index_] += 1;
            it.update();
        }
    }
}


void Array::sync_from_gpu(float * gpu_pdata, cudaStream_t stream) {
    // GPU stride array
    std::vector<unsigned int> gpu_strides_ = contiguous_strides(this->dims_, sizeof(float));

    // longest cntiguous segment and break index
    unsigned int longest_contiguous_segment_;
    int break_index_;
    std::tie(longest_contiguous_segment_, break_index_) = lcseg_and_brindex(this->dims_,
                                                                            this->strides_);

    // copy data from GPU
    if (break_index_ == -1) {  // original array is perfectly contiguous
        cudaMemcpy(this->data_, gpu_pdata,
                   longest_contiguous_segment_, cudaMemcpyDeviceToHost);
    } else {  // copy each longest_contiguous_segment
        unsigned int cpu_leap = 0;
        unsigned int gpu_leap = 0;
        for (Array::iterator it = this->begin(); it != this->end();) {
            gpu_leap = leap(it.index(), gpu_strides_);
            uintptr_t src_ptr = (uintptr_t) gpu_pdata + gpu_leap;
            cpu_leap = leap(it.index(), this->strides_);
            uintptr_t des_ptr = (uintptr_t) this->data_ + cpu_leap;
            cudaError_t err_ = cudaMemcpyAsync(reinterpret_cast<float *>(des_ptr),
                                               reinterpret_cast<float *>(src_ptr),
                                               longest_contiguous_segment_,
                                               cudaMemcpyDeviceToHost);
            if (err_ != cudaSuccess) {
                FAILURE("Memory copy failed with message \"%s\".", cudaGetErrorString(err_));
            }
            it.index()[break_index_] += 1;
            it.update();
        }
    }
}


void Array::free_data_from_gpu(int index) {
    if (index == -1) {
        while (!this->gpu_data_.empty()) {
            cudaFree(this->gpu_data_.back());
            this->gpu_data_.pop_back();
        }
    } else if ((index >= 0) && (index < this->gpu_data_.size())) {
        std::list<float *>::iterator it_ = this->gpu_data_.begin();
        std::advance(it_, index);
        this->gpu_data_.erase(it_);
    } else {
        FAILURE("Index must be in range [-1; %d], got %d.", this->gpu_data_.size()-1);
    }
}

}  // namespace merlin
