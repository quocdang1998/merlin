// Copyright 2022 quocdang1998
#include "merlin/array.hpp"

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

Array::iterator& Array::iterator::operator++(void) {
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

Array::Array(float * data, unsigned int ndim, unsigned int * dims,
             unsigned int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->dims_ = std::vector<unsigned int>(dims, dims + ndim);
    this->strides_ = std::vector<unsigned int>(strides, strides + ndim);
    this->is_copy = copy;

    // create begin and end iterator
    this->begin_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_[0] = this->dims_[0];

    // longest contiguous segment and break index
    std::vector<unsigned int> contiguous_strides(this->ndim_);
    contiguous_strides[this->ndim_ - 1] = sizeof(float);
    for (int i = ndim - 2; i >= 0; i--) {
        contiguous_strides[i] = contiguous_strides[i + 1] * this->dims_[i + 1];
    }
    this->longest_contiguous_segment_ = sizeof(float);
    this->break_index_ = ndim - 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (this->strides_[i] == contiguous_strides[i]) {
            this->longest_contiguous_segment_ *= dims[i];
            this->break_index_--;
        } else {
            break;
        }
    }

    // copy / assign data
    if (is_copy) {  // copy data
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
        for (int i = ndim - 1; i >= 0; i--) {
            this->longest_contiguous_segment_ *= dims[i];
        }
        this->break_index_ = -1;
    } else {
        this->data_ = data;
    }
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

}  // namespace merlin
