// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <algorithm>  // std::all_of
#include <cmath>      // std::pow, std::sqrt
#include <random>     // std::mt19937_64
#include <sstream>    // std::ostringstream
#include <vector>     // std::vector

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/env.hpp"          // merlin::Environment
#include "merlin/logger.hpp"       // FAILURE
#include "merlin/slice.hpp"        // merlin::Slice, merlin::slicevec
#include "merlin/stat/moment.hpp"  // merlin::stat::mean_variance
#include "merlin/utils.hpp"        // merlin::ptr_to_subsequence

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from shape and rank
candy::Model::Model(const intvec & shape, std::uint64_t rank) :
rshape_(shape), rank_(rank), param_vectors_(shape.size()) {
    // check rank
    if (rank == 0) {
        FAILURE(std::invalid_argument, "Cannot initialize a canonical model with rank zero.\n");
    }
    // calculate rshape and number of parameters
    std::uint64_t num_params = 0;
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        this->rshape_[i_dim] *= rank;
        num_params += this->rshape_[i_dim];
    }
    // reserve parameter vector and calculate param_vectors
    this->parameters_ = floatvec(num_params);
    ptr_to_subsequence(this->parameters_.data(), this->rshape_, this->param_vectors_.data());
}

// Constructor from model values
candy::Model::Model(const Vector<floatvec> & param_vectors, std::uint64_t rank) :
rshape_(param_vectors.size()), rank_(rank), param_vectors_(param_vectors.size()) {
    // check rank
    if (rank == 0) {
        FAILURE(std::invalid_argument, "Cannot initialize a canonical model with rank zero.\n");
    }
    // calculate rshape and number of parameters
    std::uint64_t num_params = 0;
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (param_vectors[i_dim].size() % rank != 0) {
            FAILURE(std::invalid_argument, "Size of all canonical model vectors must be divisible by the rank.\n");
        }
        this->rshape_[i_dim] = param_vectors[i_dim].size();
        num_params += this->rshape_[i_dim];
    }
    // reserve parameter vector and calculate param_vectors
    this->parameters_ = floatvec(num_params);
    ptr_to_subsequence(this->parameters_.data(), this->rshape_, this->param_vectors_.data());
    // copy parameter values
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        for (std::uint64_t i_param = 0; i_param < this->rshape_[i_dim]; i_param++) {
            this->param_vectors_[i_dim][i_param] = param_vectors[i_dim][i_param];
        }
    }
}

// Copy constructor
candy::Model::Model(const candy::Model & src) :
parameters_(src.parameters_), rshape_(src.rshape_), rank_(src.rank_), param_vectors_(src.ndim()) {
    ptr_to_subsequence(this->parameters_.data(), this->rshape_, this->param_vectors_.data());
}

// Copy assignment
candy::Model & candy::Model::operator=(const candy::Model & src) {
    this->parameters_ = src.parameters_;
    this->rshape_ = src.rshape_;
    this->rank_ = src.rank_;
    this->param_vectors_ = Vector<double *>(src.ndim());
    ptr_to_subsequence(this->parameters_.data(), this->rshape_, this->param_vectors_.data());
    return *this;
}

// Initialize values of model based on train data
void candy::Model::initialize(const array::Array & train_data, std::uint64_t n_thread) {
    // check shape
    if (this->ndim() != train_data.ndim()) {
        FAILURE(std::invalid_argument, "Expected train data having the same ndim as the model.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (train_data.shape()[i_dim] * this->rank() != this->rshape_[i_dim]) {
            FAILURE(std::invalid_argument, "Incoherent shape of data and model.\n");
        }
    }
    // initialize model parameters
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        std::uint64_t num_division = train_data.shape()[i_dim];
        slicevec slice_division(train_data.ndim());
        for (std::uint64_t i_division = 0; i_division < num_division; i_division++) {
            // calculate mean value for each hyper-slice
            slice_division[i_dim] = Slice(i_division, i_division + 1, 1);
            array::Array subset_data(train_data, slice_division);
            auto [mean, variance] = stat::mean_variance(subset_data, n_thread);
            double stdeviation = std::sqrt(variance);
            // calculate mean <- (mean/rank)^(1/ndim) and stdeviation <- stdeviation * new_mean / old_mean
            stdeviation /= mean;
            mean = std::pow(mean / this->rank_, 1.f / this->ndim());
            stdeviation *= mean;
            std::normal_distribution<double> generator(mean, stdeviation);
            // initialize by random
            for (std::uint64_t r = 0; r < this->rank(); r++) {
                double & param = this->get(i_dim, i_division, r);
                do {
                    param = generator(Environment::random_generator);
                } while (param <= 0);
            }
        }
    }
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * candy::Model::copy_to_gpu(candy::Model * gpu_ptr, void * grid_vector_data_ptr, std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

// Copy data from GPU to CPU
void * candy::Model::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// Check if these is a negative parameter in the model
bool candy::Model::check_negative(void) const noexcept {
    return std::all_of(this->parameters_.cbegin(), this->parameters_.cend(),
                       [](const double & value) { return value >= 0; });
}

// String representation
std::string candy::Model::str(void) const {
    std::ostringstream out_stream;
    out_stream << "<Model(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        out_stream << ((i_dim != 0) ? " " : "");
        out_stream << "<";
        std::uint64_t dim_shape = this->rshape_[i_dim] / this->rank_;
        for (std::uint64_t i_index = 0; i_index < dim_shape; i_index++) {
            floatvec rank_vector;
            double * rank_vector_data = this->param_vectors_[i_dim] + i_index * this->rank_;
            rank_vector.assign(rank_vector_data, this->rank_);
            out_stream << ((i_index != 0) ? " " : "");
            out_stream << rank_vector.str();
        }
        out_stream << ">";
    }
    out_stream << ")>";
    return out_stream.str();
}

// Destructor
candy::Model::~Model(void) {}

}  // namespace merlin
