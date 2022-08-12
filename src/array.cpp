// Copyright 2022 quocdang1998
#include "merlin/tensor.hpp"

#include <cstring>
#include <tuple>

#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Tensor::iterator
// --------------------------------------------------------------------------------------------------------------------

bool operator!= (const Tensor::iterator & left, const Tensor::iterator & right) {
    // check if 2 iterators comes from the same tensor
    if (left.dims_ != right.dims_) {
        FAILURE("2 iterators are not comming from the same tensor.");
    }

    // compare index of each iterator
    unsigned int length = left.index().size();
    for (int i = 0; i < length; i++) {
        if (left.index_[i] != right.index_[i]) {
            return true;
        }
    }
    return false;
}


void Tensor::iterator::update(void) {
    // detect dimensions having index bigger than dim
    unsigned int current_dim = this->index_.size();
    std::vector<unsigned int>& dims = this->dims();
    for (int i = this->index_.size() - 1; i >= 0; i--) {
        if (this->index_[i] >= dims[i]) {
            current_dim = i;
            break;
        }
    }
    if (current_dim == this->index_.size()) {  // no update needed
        return;
    }

    // carry the surplus to the dimensions with bigger strides
    while (this->index_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == dims[current_dim]) {
                break;
            } else {
                FAILURE("Maximum size reached, cannot add more.");
            }
        }
        div_t carry = div(static_cast<int>(this->index_[current_dim]), static_cast<int>(dims[current_dim]));
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
}


Tensor::iterator & Tensor::iterator::operator++(void) {
    this->index_[this->index_.size() - 1]++;
    unsigned int current_dim = this->index_.size() - 1;
    std::vector<unsigned int>& dims = this->dims();
    while (this->index_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == dims[current_dim]) {
                break;
            } else {
                FAILURE("Maximum size reached, cannot add more.");
            }
        }
        this->index_[current_dim] = 0;
        this->index_[--current_dim] += 1;
    }
    return *this;
}


Tensor::iterator Tensor::iterator::operator++(int) {
    return ++(*this);
}

// --------------------------------------------------------------------------------------------------------------------
// Tensor (CPU)
// --------------------------------------------------------------------------------------------------------------------

// Tools
// -----

void Tensor::contiguous_copy_from_address_(float * src, const unsigned int * src_strides) {
    // longest cntiguous segment and break index
    unsigned int longest_contiguous_segment_;
    int break_index_;
    std::vector<unsigned int> src_strides_vec(src_strides, src_strides + this->ndim_);
    std::tie(longest_contiguous_segment_, break_index_) = lcseg_and_brindex(this->dims_, src_strides_vec);

    if (break_index_ == -1) {  // original tensor is perfectly contiguous
        std::memcpy(this->data_, src, longest_contiguous_segment_);
    } else {  // memcpy each longest_contiguous_segment
        unsigned int src_leap = 0;
        unsigned int des_leap = 0;

        for (Tensor::iterator it = this->begin(); it != this->end();) {
            src_leap = leap(it.index(), src_strides_vec);
            uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src) + src_leap;
            des_leap = leap(it.index(), this->strides_);
            uintptr_t des_ptr = reinterpret_cast<uintptr_t>(this->data_) + des_leap;
            std::memcpy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr),
                        longest_contiguous_segment_);
            it.index()[break_index_] += 1;
            it.update();
        }
    }
}

// Constructors
// ------------

Tensor::Tensor(float value) {
    // allocate data
    this->data_ = new float[1];
    this->data_[0] = value;

    // set metadata
    this->ndim_ = 1;
    this->strides_ = std::vector<unsigned int>(1, sizeof(float));
    this->dims_ = std::vector<unsigned int>(1, 1);
    this->force_free = true;
}


Tensor::Tensor(const std::vector<unsigned int> & dims) {
    // initialize dims ans ndim
    this->ndim_ = dims.size();
    this->dims_ = dims;

    // initialize stride
    this->strides_ = contiguous_strides(dims, sizeof(float));

    // initialize data
    this->data_ = new float[this->size()];

    // other meta data
    this->force_free = true;
}


Tensor::Tensor(float * data, unsigned int ndim, unsigned int * dims, unsigned int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->dims_ = std::vector<unsigned int>(dims, dims + ndim);
    this->strides_ = std::vector<unsigned int>(strides, strides + ndim);
    this->force_free = copy;

    // copy / assign data
    if (copy) {  // copy data
        // allocate a new tensor
        this->data_ = new float[this->size()];

        // reform the stride tensor (force into C shape)
        this->strides_ = contiguous_strides(this->dims_, sizeof(float));

        // copy data from old tensor to new tensor (optimized with memcpy)
        this->contiguous_copy_from_address_(data, strides);
    } else {
        this->data_ = data;
    }
}


Tensor::Tensor(const Tensor & src) {
    // copy meta data
    this->ndim_ = src.ndim_;
    this->dims_ = src.dims_;
    this->strides_ = contiguous_strides(this->dims_, sizeof(float));
    this->force_free = true;

    // copy data
    this->data_ = new float[this->size()];
    this->contiguous_copy_from_address_(src.data_, &(src.strides_[0]));
}


Tensor & Tensor::operator=(const Tensor & src) {
    // copy meta data
    this->ndim_ = src.ndim_;
    this->dims_ = src.dims_;
    this->strides_ = contiguous_strides(this->dims_, sizeof(float));
    this->force_free = true;

    // copy data
    this->data_ = new float[this->size()];
    this->contiguous_copy_from_address_(src.data_, &(src.strides_[0]));

    return *this;
}


Tensor::Tensor(Tensor && src) {
    // move meta data
    this->ndim_ = src.ndim_;
    this->dims_ = std::move(src.dims_);
    this->strides_ = std::move(src.strides_);

    // disable force_free of the source
    this->force_free = src.force_free;
    src.force_free = false;

    // move data
    this->data_ = src.data_;
    src.data_ = NULL;
    this->gpu_data_ = src.gpu_data_;
}


Tensor & Tensor::operator=(Tensor && src) {
    // move meta data
    this->ndim_ = src.ndim_;
    this->dims_ = std::move(src.dims_);
    this->strides_ = std::move(src.strides_);

    // disable force_free of the source and free current data
    if (this->force_free) {
        delete[] this->data_;
    }
    this->force_free = src.force_free;
    src.force_free = false;

    // move data
    this->data_ = src.data_;
    src.data_ = NULL;
    this->gpu_data_ = src.gpu_data_;

    return *this;
}

// Get members
// -----------

float & Tensor::operator[] (const std::vector<unsigned int> & index) {
    unsigned int leap_ = leap(index, this->strides_);
    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(this->data_) + leap_;
    return *(reinterpret_cast<float *>(data_ptr));
}


unsigned int Tensor::size(void) {
    unsigned int size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->dims_[i];
    }
    return size;
}

// Iterator
// --------

Tensor::iterator Tensor::begin(void) {
    this->begin_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_[0] = this->dims_[0];
    return Tensor::iterator(this->begin_, this->dims_);
}


Tensor::iterator Tensor::end(void) {
    return Tensor::iterator(this->end_, this->dims_);
}

// Diable GPU features
#ifndef __NVCC__
void Tensor::sync_to_gpu(float * gpu_pdata, uintptr_t stream) {
    FAILURE("Merlin is not compiled with nvcc.");
}


void Tensor::sync_from_gpu(float * gpu_pdata, uintptr_t stream) {
    FAILURE("Merlin is not compiled with nvcc.");
}
#endif  // __NVCC__

// Destructor
// ----------

Tensor::~Tensor(void) noexcept(false) {
    if (this->force_free) {
        // MESSAGE("Free CPU data.");
        delete[] this->data_;
    }
    if (this->gpu_data_.size() > 0) {
        FAILURE("GPU data left unfreed!");
    }
}

}  // namespace merlin
