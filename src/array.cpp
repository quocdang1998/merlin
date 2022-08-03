// Copyright 2022 quocdang1998
#include "merlin/array.hpp"

#include <cstdio>

namespace merlin {

// ------------------------------------------------------------------------------------------------
// Array::iterator
// ------------------------------------------------------------------------------------------------

bool operator!= (const Array::iterator& left, const Array::iterator& right) {
    // check if 2 iterators comes from the same array
    if (left.dims_ != right.dims_) {
        throw(std::runtime_error("2 iterators are not comming from the same array."));
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

void Array::iterator::update(void) {
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
                throw(std::out_of_range("Maximum size reached, cannot add more."));
            }
        }
        div_t carry = div(static_cast<int>(this->index_[current_dim]),
                          static_cast<int>(dims[current_dim]));
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
}

Array::iterator & Array::iterator::operator++(void) {
    this->index_[this->index_.size() - 1]++;
    unsigned int current_dim = this->index_.size() - 1;
    std::vector<unsigned int>& dims = this->dims();
    while (this->index_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == dims[current_dim]) {
                break;
            } else {
                throw(std::out_of_range("Maximum size reached, cannot add more."));
            }
        }
        this->index_[current_dim] = 0;
        this->index_[--current_dim] += 1;
    }
    return *this;
}

Array::iterator Array::iterator::operator++(int) {
    return ++(*this);
}

// ------------------------------------------------------------------------------------------------
// Array (CPU)
// ------------------------------------------------------------------------------------------------

Array::Array(float value) {
    this->data_ = new float[1];
    this->data_[0] = value;
    this->ndim_ = 1;
    this->strides_ = std::vector<unsigned int>(1, sizeof(float));
    this->dims_ = std::vector<unsigned int>(1, 1);
    this->begin_ = std::vector<unsigned int>(1, 0);
    this->end_ = std::vector<unsigned int>(1, 1);
    this->force_free = true;
    this->longest_contiguous_segment_ = sizeof(float);
    this->break_index_ = -1;
}

Array::Array(const std::vector<unsigned int> & dims) {
    // initialize dims ans ndim
    this->ndim_ = dims.size();
    this->dims_ = dims;

    // initialize stride
    this->strides_ = std::vector<unsigned int>(dims.size(), sizeof(float));
    for (int i = dims.size()-2; i >= 0; i--) {
        this->strides_[i] = this->strides_[i+1] * dims[i+1];
    }

    // initialize data
    this->data_ = new float[this->size()];

    // other meta data
    this->begin_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_[0] = this->dims_[0];
    this->longest_contiguous_segment_ = sizeof(float);
    for (int i = this->ndim_-1; i >= 0; i--) {
        this->longest_contiguous_segment_ *= dims[i];
    }
    this->break_index_ = -1;
    this->force_free = true;
}

Array::Array(float * data, unsigned int ndim, unsigned int * dims,
             unsigned int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->dims_ = std::vector<unsigned int>(dims, dims + ndim);
    this->strides_ = std::vector<unsigned int>(strides, strides + ndim);
    this->force_free = copy;

    // create begin and end iterator
    this->begin_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_[0] = this->dims_[0];

    // longest contiguous segment and break index
    std::vector<unsigned int> contiguous_strides(this->ndim_);
    contiguous_strides[this->ndim_ - 1] = sizeof(float);
    for (int i = ndim-2; i >= 0; i--) {
        contiguous_strides[i] = contiguous_strides[i+1] * this->dims_[i+1];
    }
    this->longest_contiguous_segment_ = sizeof(float);
    this->break_index_ = ndim - 1;
    for (int i = ndim-1; i >= 0; i--) {
        if (this->strides_[i] == contiguous_strides[i]) {
            this->longest_contiguous_segment_ *= dims[i];
            this->break_index_--;
        } else {
            break;
        }
    }

    // copy / assign data
    if (copy) {  // copy data
        // allocate a new array
        this->data_ = new float[this->size()];

        // reform the stride array (force into C shape)
        this->strides_ = contiguous_strides;

        // copy data from old array to new array (optimized with memcpy)
        if (this->break_index_ == -1) {  // original array is perfectly contiguous
            std::memcpy(this->data_, data, this->longest_contiguous_segment_);
        } else {  // memcpy each longest_contiguous_segment
            for (Array::iterator it = this->begin(); it != this->end();) {
                unsigned int leap = 0;
                for (int i = 0; i < it.index().size(); i++) {
                    leap += it.index()[i] * strides[i];
                }
                uintptr_t src_ptr = reinterpret_cast<uintptr_t>(data) + leap;
                uintptr_t des_ptr = reinterpret_cast<uintptr_t>(&(this->operator[](it.index())));
                std::memcpy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr),
                    this->longest_contiguous_segment_);
                it.index()[this->break_index_] += 1;
                try {
                    it.update();
                }
                catch (const std::out_of_range& err) {
                    break;
                }
            }
        }

        // reset longest contiguous segment and break index
        this->longest_contiguous_segment_ = sizeof(float);
        for (int i = ndim-1; i >= 0; i--) {
            this->longest_contiguous_segment_ *= dims[i];
        }
        this->break_index_ = -1;
    } else {
        this->data_ = data;
    }
}

Array::Array(const Array & src) {
    // copy meta data
    this->ndim_ = src.ndim_;
    this->dims_ = src.dims_;
    this->begin_ = src.begin_;
    this->end_ = src.end_;
    this->force_free = true;

    // strides
    for (int i = this->ndim_-2; i >= 0; i--) {
        this->strides_[i] = this->strides_[i+1] * this->dims_[i+1];
    }

    // copy data
    this->data_ = new float[this->size()];
    if (src.break_index_ == -1) {
        std::memcpy(this->data_, src.data_, src.longest_contiguous_segment_);
    } else {
        for (Array::iterator it = this->begin(); it != this->end();) {
            unsigned int leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                leap += it.index()[i] * src.strides_[i];
            }
            uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src.data_) + leap;
            uintptr_t des_ptr = reinterpret_cast<uintptr_t>(&(this->operator[](it.index())));
            std::memcpy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr),
                        src.longest_contiguous_segment_);
            it.index()[this->break_index_] += 1;
            try {
                it.update();
            }
            catch (const std::out_of_range& err) {
                break;
            }
        }
    }

    // break index and longest contiguous segment
    this->longest_contiguous_segment_ = sizeof(float);
    for (int i = this->ndim_-1; i >= 0; i--) {
        this->longest_contiguous_segment_ *= this->dims_[i];
    }
    this->break_index_ = -1;
}

Array & Array::operator=(const Array & src) {
    // copy meta data
    this->ndim_ = src.ndim_;
    this->dims_ = src.dims_;
    this->begin_ = src.begin_;
    this->end_ = src.end_;
    this->force_free = true;

    // strides
    this->strides_ = std::vector<unsigned int>(this->ndim_, sizeof(float));
    for (int i = this->ndim_-2; i >= 0; i--) {
        this->strides_[i] = this->strides_[i+1] * this->dims_[i+1];
    }

    // copy data
    this->data_ = new float[this->size()];
    if (src.break_index_ == -1) {
        std::memcpy(this->data_, src.data_, src.longest_contiguous_segment_);
    } else {
        for (Array::iterator it = this->begin(); it != this->end();) {
            unsigned int leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                leap += it.index()[i] * src.strides_[i];
            }
            uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src.data_) + leap;
            uintptr_t des_ptr = reinterpret_cast<uintptr_t>(&(this->operator[](it.index())));
            std::memcpy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr),
                        src.longest_contiguous_segment_);
            it.index()[this->break_index_] += 1;
            try {
                it.update();
            }
            catch (const std::out_of_range& err) {
                break;
            }
        }
    }

    // break index and longest contiguous segment
    this->longest_contiguous_segment_ = sizeof(float);
    for (int i = this->ndim_-1; i >= 0; i--) {
        this->longest_contiguous_segment_ *= this->dims_[i];
    }
    this->break_index_ = -1;

    return *this;
}

Array::Array(Array && src) {
    // move meta data
    this->ndim_ = src.ndim_;
    this->dims_ = std::move(src.dims_);
    this->begin_ = std::move(src.begin_);
    this->end_ = std::move(src.end_);
    this->longest_contiguous_segment_ = src.longest_contiguous_segment_;
    this->break_index_ = src.break_index_;

    // disable force_free of the source
    this->force_free = src.force_free;
    src.force_free = false;

    // move data
    this->data_ = src.data_;
    src.data_ = NULL;
    this->gpu_data_ = src.gpu_data_;
    src.gpu_data_ = NULL;
}

float & Array::operator[] (const std::vector<unsigned int> & index) {
    unsigned int leap = 0;
    if (index.size() != this->ndim_) {
        throw std::length_error("Index must have the same length as array");
    }
    for (int i = 0; i < index.size(); i++) {
        leap += index[i] * this->strides_[i];
    }
    uintptr_t data_ptr = (uintptr_t)this->data_ + leap;
    return *(reinterpret_cast<float *>(data_ptr));
}

unsigned int Array::size(void) {
    unsigned int size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->dims_[i];
    }
    return size;
}

Array::iterator Array::begin(void) {
    return Array::iterator(this->begin_, this->dims_);
}

Array::iterator Array::end(void) {
    return Array::iterator(this->end_, this->dims_);
}

Array::~Array(void) {
    // free CPU data if copy enabled
    if (this->force_free) {
        std::printf("Free CPU data.\n");
        delete[] this->data_;
    }
#ifdef __NVCC__
    // free GPU data
    if (this->gpu_data_ != NULL) {
        std::printf("Free GPU data.\n");
        cudaFree(this->gpu_data_);
    }
#endif  // __NVCC__
}

}  // namespace merlin
