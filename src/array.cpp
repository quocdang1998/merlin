#include "merlin/array.hpp"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <iostream>

namespace merlin {

// for debug purpose only
static void print_vec(const merlin::Array::iterator & it) {
    const std::vector<unsigned int> & v = it.it();
    for (const unsigned int & i : v) {
        std::printf("%d ", i);
    }
    std::printf("\n");
}

void Array::iterator::update(void) {
    // detect dimensions having index bigger than dim
    unsigned int current_dim = this->it_.size();
    std::vector<unsigned int> & dims = this->dims();
    for (int i = this->it_.size() - 1; i >=0; i--) {
        if (this->it_[i] >= dims[i]) {
            current_dim = i;
            break;
        }
    }
    if (current_dim == this->it_.size()) {  // no update needed
        return;
    }

    // carry the surplus to the dimensions with bigger strides
    while (this->it_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->it_[current_dim] == dims[current_dim]) {
                break;
            }
            else {
                throw(std::out_of_range("Maximum size reached, cannot add more."));
            }
        }
        div_t carry = div((int) this->it_[current_dim], (int) dims[current_dim]);
        this->it_[current_dim] = carry.rem;
        this->it_[--current_dim] += carry.quot;
    }
}

bool operator!= (const Array::iterator & left, const Array::iterator & right) {
    // check if 2 iterators comes from the same array
    if (left.dims_ != right.dims_) {
        throw(std::runtime_error("2 iterators are not comming from the same array."));
    }

    // compare index of each iterator
    unsigned int length = left.it().size();
    for (int i = 0; i < length; i++) {
        if (left.it_[i] != right.it_[i]) {
            return true;
        }
    }
    return false;
}

Array::iterator& Array::iterator::operator++(void) {
    this->it_[this->it_.size()-1]++;
    unsigned int current_dim = this->it_.size()-1;
    std::vector<unsigned int> & dims = this->dims();
    while (this->it_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->it_[current_dim] == dims[current_dim]) {
                break;
            }
            else {
                throw(std::out_of_range("Maximum size reached, cannot add more."));
            }
        }
        this->it_[current_dim] = 0;
        this->it_[--current_dim] += 1;
    }
    return *this;
}

Array::iterator Array::iterator::operator++(int) {
    return ++(*this);
}

Array::Array(double * data, unsigned int ndim, unsigned int * dims,
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

    // copy / assign data
    if (is_copy) {  // copy data
        // allocate a new array
        this->data_ = new double [this->size()];

        // reform the stride array (force into C shape)
        this->strides_[ndim-1] = sizeof(double);
        for (int i = ndim-2; i >= 0; i--) {
            this->strides_[i] = this->strides_[i+1] * this->dims_[i+1];
        }

        // longest contiguous array
        unsigned int longest_contiguous_segment = sizeof(double);
        int break_index = ndim-1;
        for (int i = ndim-1; i >=0; i--) {
            if (this->strides_[i] == strides[i]) {
                longest_contiguous_segment *= dims[i];
                break_index--;
            } else {
                break;
            }
        }

        // copy data from old array to new array (optimized with memcpy)
        if (break_index == -1) {  // original array is perfectly contiguous
            std::memcpy(this->data_, data, longest_contiguous_segment);
        } else {  // memcpy each longest_contiguous_segment
            for (Array::iterator it = this->begin(); it != this->end();) {
                unsigned int leap = 0;
                for (int i = 0; i < it.it().size(); i++) {
                    leap += it.it()[i] * strides[i];
                }
                uintptr_t src_ptr = (uintptr_t) data + leap;
                uintptr_t des_ptr = (uintptr_t) &(this->operator[](it.it()));
                std::memcpy((double *) des_ptr, (double *) src_ptr,
                            longest_contiguous_segment);
                it.it()[break_index] += 1;
                try {
                    it.update();
                } catch (const std::out_of_range & err) {
                    break;
                }
            }
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
    return Array::iterator(this->begin_, this->dims_);
}

Array::iterator Array::end(void) {
    return Array::iterator(this->end_, this->dims_);
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
