// Copyright 2022 quocdang1998
#include "merlin/nddata.hpp"

#include <cstdlib>  // std::abs

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// NdData
// -------------------------------------------------------------------------------------------------------------------------

// Member initialization for C++ interface
NdData::NdData(float * data, unsigned long int ndim, std::initializer_list<unsigned long int> shape,
             std::initializer_list<unsigned long int> strides) : ndim_(ndim), shape_(shape), strides_(strides) {
    this->data_ = data;
    if (shape.size() != ndim) {
        FAILURE(std::range_error, "Expected size of shape (%u) equals to ndim (%u).\n", shape.size(), ndim);
    } else if (strides.size() != ndim) {
        FAILURE(std::range_error, "Expected size of strides (%u) equals to ndim (%u).\n", strides.size(), ndim);
    }
}

// Constructor from Numpy np.array
NdData::NdData(float * data, unsigned long int ndim, const unsigned long int * shape, const unsigned long int * strides) {
    this->ndim_ = ndim;
    this->data_ = data;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
}

// Number of elements
unsigned long int NdData::size(void) {
    unsigned long int size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->shape_[i];
    }
    return size;
}

#ifdef FUTURE
// -------------------------------------------------------------------------------------------------------------------------
// Slice
// -------------------------------------------------------------------------------------------------------------------------

// Constructor from members
Slice::Slice(unsigned int start, int stop, int step) : start_(start), stop_(stop), step_(step) {
    if (step == 0) {
        FAILURE(std::invalid_argument, "Step must not be 0.\n");
    }
}

// Constructor from initializer list
Slice::Slice(std::initializer_list<int> list) {
    // check size of initializer_list
    if (list.size() != 3) {
        FAILURE(std::invalid_argument, "Expected intializer list with size 3, got %d.\n", list.size());
    }
    // check if first element is positive
    const int * list_data = std::data(list);
    if (list_data[0] < 0) {
        FAILURE(std::invalid_argument, "First element (%d) must be a positive number.\n", list_data[0]);
    }
    // assign each element to object members
    this->start_ = list_data[0];
    this->stop_ = list_data[1];
    this->step_ = list_data[2];
}

// Conver Slice to vector of corresponding indices
std::vector<unsigned int> Slice::range(unsigned int length) {
    // check validity of start and stop
    if (this->start_ > length) {
        FAILURE(std::length_error, "Start index (%u) bigger than length (%u).\n", this->start_, length);
    } else if (std::abs(this->stop_) > length) {
        FAILURE(std::length_error, "Stop index (%d) must be in range [%u, %u].\n", this->stop_, length, length);
    }
    // intialize range to be returned
    std::vector<unsigned int> range;
    // positive step (step check for zero has been done in constructor)
    if (this->step_ > 0) {
        // get mod(stop_index, length) and convert it in range [1, ..., length]
        unsigned int stop_index = this->stop_ % length;
        // loop and save indices
        for (unsigned int i = this->start_; i < stop_index; i += this->step_) {
            range.push_back(i);
        }
    } else {  // negative step
        // get mod(stop_index, length) and convert start_index in range [1, ..., length]
        unsigned int stop_index = this->stop_ % length;
        unsigned int start_index = this->start_;
        if (start_index == 0) {
            start_index = length;
        }
        // loop and save indices
        for (unsigned int i = this->start_; i > stop_index; i += this->step_) {
            range.push_back(i);
        }
    }
    return range;
}
#endif

}  // namespace merlin
