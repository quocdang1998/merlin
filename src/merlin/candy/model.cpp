// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <cmath>  // std::pow, std::sqrt
#include <cstring>  // std::memcpy
#include <sstream>  // std::ostringstream
#include <random>  // std::mt19937_64, std::uniform_real_distribution
#include <utility>  // std::exchange
#include <vector>  // std::vector

#include <omp.h>  // pragma omp, ::omp_get_threa

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/slice.hpp"  // merlin::Slice, merlin::slicevec
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/statistics/moment.hpp"  // merlin::statistics::powered_mean, merlin::statistics::moment_cpu
                                         // merlin::statistics::max_cpu
#include "merlin/utils.hpp"  // merlin::prod_elements

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Initialize model
// --------------------------------------------------------------------------------------------------------------------

// Initialize a random engine to each thread
static std::vector<std::mt19937_64> initialize_engine(std::uint64_t n_thread) {
    std::vector<std::mt19937_64> engines(n_thread);
    for (std::uint64_t i_thread = 0; i_thread < n_thread; i_thread++) {
        engines[i_thread] = std::mt19937_64(Environment::random_generator());
    }
    return engines;
}

// Initialize value by default distribution
static void initialize_default(candy::Model & model, std::uint64_t n_thread) {
    std::vector<std::mt19937_64> engines = initialize_engine(n_thread);
    std::uniform_real_distribution<double> generator(-1.0, 1.0);
    std::uint64_t size = model.size();
    #pragma omp parallel for num_threads(n_thread)
    for (std::int64_t i_param = 0; i_param < size; i_param++) {
        double generated_value = 0.0;
        do {
            generated_value = generator(engines[::omp_get_thread_num()]);
        } while (generated_value == 0);
        model.set(i_param, generated_value);
    }
}

// Initialize value by normal distribution
static void initialize_gaussian(candy::Model & model, const array::Array & train_data, std::uint64_t n_thread) {
    // initialize distribution
    std::vector<std::mt19937_64> engines = initialize_engine(n_thread);
    // loop over each dimension
    for (std::uint64_t i_dim = 0; i_dim < model.ndim(); i_dim++) {
        // initialize parameter vectors
        std::uint64_t num_division = train_data.shape()[i_dim];
        floatvec mean_division(num_division);
        // loop over each hyper-slice on the dimension
        for (std::uint64_t i_division = 0; i_division < num_division; i_division++) {
            // calculate mean value for each hyper-slice
            slicevec slice_division(train_data.ndim());
            slice_division[i_dim] = array::Slice({static_cast<std::uint64_t>(i_division)});
            array::Array subset_data(train_data, slice_division);
            std::array<double, 2> means = statistics::powered_mean<2>(subset_data, n_thread);
            double deviation = statistics::moment_cpu<2>(means);
            deviation = std::sqrt(deviation);
            double mean = means[0] / model.rank();
            mean = std::pow(mean, 1.f / train_data.ndim());
            std::normal_distribution<double> generator(mean, deviation);
            for (std::uint64_t r = 0; r < model.rank(); r++) {
                double & param = model.parameters()[i_dim][i_division * model.rank() + r];
                do {
                    param = generator(engines[::omp_get_thread_num()]);
                } while (param == 0);
            }
        }
    }
}

// Initialize value by uniform distribution
static void initialize_uniform(candy::Model & model, const array::Array & train_data, std::uint64_t n_thread) {
    std::vector<std::mt19937_64> engines = initialize_engine(n_thread);
    double max_value = statistics::max_cpu(train_data, n_thread);
    std::uniform_real_distribution<double> generator(-max_value, max_value);
    std::uint64_t size = model.size();
    #pragma omp parallel for num_threads(n_thread)
    for (std::int64_t i_param = 0; i_param < size; i_param++) {
        double generated_value = 0.0;
        do {
            generated_value = generator(engines[::omp_get_thread_num()]);
        } while (generated_value == 0);
        model.set(i_param, generated_value);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Model
// --------------------------------------------------------------------------------------------------------------------

// Constructor from shape and rank
candy::Model::Model(const intvec & shape, std::uint64_t rank) : parameters_(shape.size()), rank_(rank) {
    // check rank
    if (rank == 0) {
        FAILURE(std::invalid_argument, "Cannot initialize a canonical model with rank zero.\n");
    }
    // allocate data and assign a random number between 0 and 1 for each entry
    std::uniform_real_distribution<double> generator(-1.0, 1.0);
    for (std::uint64_t i_dim = 0; i_dim < shape.size(); i_dim++) {
        this->parameters_[i_dim] = floatvec(shape[i_dim]*rank);
        for (std::uint64_t i_param = 0; i_param < this->parameters_[i_dim].size(); i_param++) {
            // generate a random strictly positive floating-point value
            double assigned_value = generator(Environment::random_generator);
            while (assigned_value == 0.0) {
                assigned_value = generator(Environment::random_generator);
            }
            this->parameters_[i_dim][i_param] = assigned_value;
        }
    }
}

// Constructor from model values
candy::Model::Model(const Vector<floatvec> & parameter, std::uint64_t rank) : parameters_(parameter),
rank_(rank) {
    // check rank
    if (rank == 0) {
        FAILURE(std::invalid_argument, "Cannot initialize a canonical model with rank zero.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < parameter.size(); i_dim++) {
        if (parameter[i_dim].size() % rank != 0) {
            FAILURE(std::invalid_argument, "Size of all canonical model vectors must be divisible by the rank.\n");
        }
        for (std::uint64_t i_param = 0; i_param < parameter[i_dim].size(); i_param++) {
            if (parameter[i_dim][i_param] == 0.0) {
                FAILURE(std::invalid_argument, "Intial values of model cannot contain zeros.\n");
            }
        }
    }
}

// Slicing constructor
candy::Model::Model(Model & full_model, const slicevec & slices) : parameters_(full_model.ndim()),
rank_(full_model.rank_) {
    if (slices.size() != full_model.ndim()) {
        CUHDERR(std::invalid_argument, "Inconsistent size of Model and slice vector.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < full_model.ndim(); i_dim++) {
        if (slices[i_dim].step() != 1) {
            CUHDERR(std::invalid_argument, "Expected slice with step 1.\n");
        }
        auto [offset, shape, stride] = slices[i_dim].slice_on(full_model.parameters_[i_dim].size() / full_model.rank_,
                                                              full_model.rank_ * sizeof(double));
        this->parameters_[i_dim].assign(full_model.parameters_[i_dim].data() + offset, shape);
    }
}

// Initialize values of model based on train data
void candy::Model::initialize(const array::Array & train_data, candy::RandomInitializer random_distribution,
                              std::uint64_t n_thread) {
    // check shape
    if (this->ndim() != train_data.ndim()) {
        FAILURE(std::invalid_argument, "Expected train data having the same ndim as the model.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        if (train_data.shape()[i_dim] * this->rank() != this->parameters_[i_dim].size()) {
            FAILURE(std::invalid_argument, "Incoherent shape of data and model.\n");
        }
    }
    // initialize model parameters
    switch (random_distribution) {
    case candy::RandomInitializer::DefaultDistribution:
        initialize_default(*this, n_thread);
        break;
    case candy::RandomInitializer::UniformDistribution:
        initialize_uniform(*this, train_data, n_thread);
        break;
    case candy::RandomInitializer::NormalDistribution:
        initialize_gaussian(*this, train_data, n_thread);
        break;
    default:
        FAILURE(std::invalid_argument, "Not implemented random distribution.\n");
        break;
    }
}

// Calculate minimum size to allocate to store the object
std::uint64_t candy::Model::malloc_size(void) const {
    std::uint64_t size = sizeof(candy::Model) + this->ndim()*sizeof(floatvec);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        size += this->parameters_[i_dim].size() * sizeof(double);
    }
    return size;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * candy::Model::copy_to_gpu(candy::Model * gpu_ptr, void * grid_vector_data_ptr,
                                 std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

// Copy data from GPU to CPU
void * candy::Model::copy_from_gpu(void * parameters_data_ptr, std::uintptr_t stream_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// String representation
std::string candy::Model::str(void) const {
    std::ostringstream out_stream;
    out_stream << "<Candecomp Model(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        out_stream << ((i_dim != 0) ? " " : "");
        out_stream << "<";
        std::uint64_t dim_shape = this->parameters_[i_dim].size() / this->rank_;
        for (std::uint64_t i_index = 0; i_index < dim_shape; i_index++) {
            floatvec rank_vector;
            rank_vector.assign(const_cast<double *>(this->parameters_[i_dim].data()) + i_index*this->rank_, this->rank_);
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
