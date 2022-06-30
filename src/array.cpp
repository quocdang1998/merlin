#include "merlin/array.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <iostream>

namespace merlin {

bool operator< (const Array::iterator & left, const Array::iterator & right) {
    // check if 2 iterators comes from the same array
    if (left.dims_ != right.dims_) {
        throw(std::runtime_error("2 iterators are not comming from the same array."));
    }

    // compare index of each iterator
    unsigned int length = left.it().size();
    for (int i = 0; i < length; i++) {
        if (left.it_[i] > right.it_[i]) {
            return false;
        }
    }
    return true;
}

Array::iterator& Array::iterator::operator++(void) {
    this->it_[this->it_.size()-1]++;
    unsigned int current_dim = this->it_.size()-1;
    std::vector<unsigned int> & dims = this->dims();
    while (this->it_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            return *this;
        }
        this->it_[current_dim] = 0;
        this->it_[--current_dim] += 1;
    }
    return *this;
}

Array::iterator Array::iterator::operator++(int) {
    return ++(*this);
}

Array::Array(double * data, unsigned int ndim, unsigned int * dims, unsigned int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->dims_ = std::vector<unsigned int>(dims, dims + ndim);
    this->strides_ = std::vector<unsigned int>(strides, strides + ndim);
    this->is_copy = copy;


    // copy / assign data
    if (is_copy) {  // copy data
        // allocate a new array
        this->data_ = new double [this->size()];

        // detect if original array is fortran or c
        for (int i = 0; i < ndim; i++) {
            if (strides[i] == sizeof(double)) {
                fastest_index = i;
                break;
            }
        }

        // reform the stride array (force into C shape)
        
        this->strides_[ndim-1] = sizeof(double);
        for (int i = ndim-2; i >= 0; i--) {
            this->strides_[i] = this->strides_[i+1] * this->dims_[i+1];
        }

        // copy data from old array to new array (to be optimized with memcpy!)
        for (Array::iterator it = this->begin(); it < this->end(); ++it) {
            unsigned int leap = 0;
            for (int i = 0; i < it.it().size(); i++) {
                leap += it.it()[i] * strides_[i];
            }
            uintptr_t data_ptr = (uintptr_t) data_ + leap;
            this->operator[](it.it()) = *((double *) data_ptr);
        }
    } else {
        this->data_ = data;
    }
}

Array::~Array(void) {
    if (this->is_copy) {
        std::printf("Free copied data.\n");
        delete[] this->data_;
    }
}

unsigned int Array::size() {
    unsigned int size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->dims_[i];
    }
    return size;
}

Array::iterator Array::begin(void) {
    return Array::iterator(std::vector<unsigned int>(this->ndim_, 0), this->dims_);
}

Array::iterator Array::end(void) {
    std::vector<unsigned int> end;
    std::copy(this->dims_.begin(), this->dims_.end(), std::back_inserter(end));
    for (int i = 0; i < this->ndim_; i++) {
        end[i] -= 1;
    }
    return Array::iterator(end, this->dims_);
}

double & Array::operator[] (const std::vector<unsigned int> & index) {
    unsigned int leap = 0;
    if (index.size() != this->ndim_) {
        throw std::length_error("Index must have the same length as array");
    }
    for (int i = 0; i < index.size(); i++) {
        leap += index[i] * this->strides_[i];
    }
    uintptr_t data_ptr = (uintptr_t) this->data_ + leap;
    return *((double *) data_ptr);
}

}  // namespace merlin
