#include "merlin/array.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <iostream>

namespace merlin {

bool operator< (const Array::iterator & left, const Array::iterator & right) {
    unsigned int length = left.it_.size();
    for (int i = 0; i < length; i++) {
        if (left.it_[i] > right.it_[i]) {
            return false;
        }
    }
    return true;
}

void Array::iterator::inc(const std::vector<unsigned int> & dims) {
    this->it_[this->it_.size()-1]++;
    unsigned int current_dim = this->it_.size()-1;
    while (this->it_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            return;
        }
        this->it_[current_dim] = 0;
        this->it_[--current_dim] += 1;
    }
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

        // reform the stride array
        this->strides_[ndim-1] = sizeof(double);
        for (int i = ndim-2; i >= 0; i--) {
            this->strides_[i] = this->strides_[i+1] * this->dims_[i+1];
        }

        // copy data from old array to new array
        for (Array::iterator it = this->begin(); it < this->end(); it.inc(this->dims_)) {
            unsigned int leap = 0;
            for (int i = 0; i < it.it_.size(); i++) {
                leap += it.it_[i] * strides_[i];
            }
            uintptr_t data_ptr = (uintptr_t) data_ + leap;
            this->operator[](it.it_) = *((double *) data_ptr);
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
    return Array::iterator(std::vector<unsigned int>(this->ndim_, 0));
}

Array::iterator Array::end(void) {
    std::vector<unsigned int> end;
    std::copy(this->dims_.begin(), this->dims_.end(), std::back_inserter(end));
    for (int i = 0; i < this->ndim_; i++) {
        end[i] -= 1;
    }
    return Array::iterator(end);
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
