// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <algorithm>  // std::all_of, std::find
#include <cmath>      // std::pow, std::sqrt
#include <iterator>   // std::distance
#include <random>     // std::mt19937_64, std::normal_distribution, std::uniform_real_distribution
#include <sstream>    // std::ostringstream
#include <vector>     // std::vector

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/env.hpp"          // merlin::Environment
#include "merlin/filelock.hpp"     // merlin::FileLock
#include "merlin/logger.hpp"       // merlin::Fatal
#include "merlin/slice.hpp"        // merlin::SliceArray
#include "merlin/utils.hpp"        // merlin::ptr_to_subsequence

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from shape and rank
candy::Model::Model(const Index & shape, std::uint64_t rank) : rshape_(shape), rank_(rank) {
    // check rank
    if (rank == 0) {
        Fatal<std::invalid_argument>("Cannot initialize a canonical model with rank zero.\n");
    }
    // get ndim
    Index::const_iterator first_zero_element = std::find(shape.begin(), shape.end(), 0);
    this->ndim_ = std::distance(shape.begin(), first_zero_element);
    // calculate rshape and number of parameters
    std::uint64_t num_params = 0;
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        this->rshape_[i_dim] *= rank;
        num_params += this->rshape_[i_dim];
    }
    // reserve parameter vector and calculate param_vectors
    this->parameters_ = DoubleVec(num_params);
    ptr_to_subsequence(this->parameters_.data(), this->rshape_.data(), this->ndim_, this->param_vectors_.data());
}

// Constructor from model values
candy::Model::Model(const Vector<DoubleVec> & param_vectors, std::uint64_t rank) :
ndim_(param_vectors.size()), rank_(rank) {
    // check rank
    if (rank == 0) {
        Fatal<std::invalid_argument>("Cannot initialize a canonical model with rank zero.\n");
    }
    // calculate rshape and number of parameters
    if (param_vectors.size() > max_dim) {
        Fatal<std::invalid_argument>("Exceeding max dimension.\n");
    }
    std::uint64_t num_params = 0;
    this->rshape_.fill(0);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        if (param_vectors[i_dim].size() % rank != 0) {
            Fatal<std::invalid_argument>("Size of all canonical model vectors must be divisible by the rank.\n");
        }
        this->rshape_[i_dim] = param_vectors[i_dim].size();
        num_params += this->rshape_[i_dim];
    }
    // reserve parameter vector and calculate param_vectors
    this->parameters_ = DoubleVec(num_params);
    ptr_to_subsequence(this->parameters_.data(), this->rshape_.data(), this->ndim_, this->param_vectors_.data());
    // copy parameter values
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        for (std::uint64_t i_param = 0; i_param < this->rshape_[i_dim]; i_param++) {
            this->param_vectors_[i_dim][i_param] = param_vectors[i_dim][i_param];
        }
    }
}

// Copy constructor
candy::Model::Model(const candy::Model & src) :
parameters_(src.parameters_), rshape_(src.rshape_), ndim_(src.ndim_), rank_(src.rank_) {
    ptr_to_subsequence(this->parameters_.data(), this->rshape_.data(), this->ndim_, this->param_vectors_.data());
}

// Copy assignment
candy::Model & candy::Model::operator=(const candy::Model & src) {
    this->parameters_ = src.parameters_;
    this->rshape_ = src.rshape_;
    this->ndim_ = src.ndim_;
    this->rank_ = src.rank_;
    ptr_to_subsequence(this->parameters_.data(), this->rshape_.data(), this->ndim_, this->param_vectors_.data());
    return *this;
}

// Initialize values of model based on train data
void candy::Model::initialize(const array::Array & train_data) {
    // check shape
    if (!this->check_compatible_shape(train_data.shape())) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // initialize model parameters
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        std::uint64_t num_division = train_data.shape()[i_dim];
        SliceArray slice_division;
        slice_division.fill(Slice());
        for (std::uint64_t i_division = 0; i_division < num_division; i_division++) {
            // calculate mean value for each hyper-slice
            slice_division[i_dim] = Slice(i_division, i_division + 1, 1);
            array::NdData * p_subset_data = train_data.sub_array(slice_division);
            auto [mean, variance] = p_subset_data->get_mean_variance();
            delete p_subset_data;
            double stdeviation = std::sqrt(variance);
            // zero mean and variance
            if ((mean == 0.0) && (stdeviation == 0.0)) {
                for (std::uint64_t r = 0; r < this->rank_; r++) {
                    this->get(i_dim, i_division, r) = 0.0;
                }
                continue;
            }
            // calculate mean <- (mean/rank)^(1/ndim) and stdeviation <- stdeviation * new_mean / old_mean
            if (mean != 0.0) {
                stdeviation /= mean;
                mean = std::pow(mean / this->rank_, 1.f / this->ndim());
                stdeviation *= mean;
            }
            std::normal_distribution<double> generator(mean, stdeviation);
            // initialize by random
            for (std::uint64_t r = 0; r < this->rank_; r++) {
                double & param = this->get(i_dim, i_division, r);
                do {
                    param = generator(Environment::random_generator);
                } while (param <= 0);
            }
        }
    }
}

// Initialize values of model based on rank-1 model
void candy::Model::initialize(const candy::Model & rank_1_model, double rtol) {
    // check argument
    if (rank_1_model.rank_ != 1) {
        Fatal<std::invalid_argument>("Model provided is not rank-1.\n");
    }
    if (this->ndim() != rank_1_model.ndim()) {
        Fatal<std::invalid_argument>("Expected rank-1 model having the same ndim as the current model.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        if (rank_1_model.rshape_[i_dim] * this->rank_ != this->rshape_[i_dim]) {
            Fatal<std::invalid_argument>("Incoherent shape of 2 models.\n");
        }
    }
    // calculate min and max in relative for each rank
    double rstep = (2.0 * rtol) / static_cast<double>(this->rank_);
    DoubleVec rtol_rank(this->rank_ + 1);
    for (std::uint64_t r = 0; r < this->rank_ + 1; r++) {
        rtol_rank[r] = 1.0 - rtol + r * rstep;
    }
    // initialize
    for (std::uint64_t i_param = 0; i_param < rank_1_model.num_params(); i_param++) {
        double param_value = rank_1_model.parameters_[i_param];
        param_value /= std::pow(this->rank_, 1.0 / static_cast<double>(this->ndim_));
        for (std::uint64_t r = 0; r < this->rank_; r++) {
            std::uniform_real_distribution<double> generator(param_value * rtol_rank[r],
                                                             param_value * rtol_rank[r + 1]);
            this->parameters_[i_param * this->rank_ + r] = generator(Environment::random_generator);
        }
    }
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * candy::Model::copy_to_gpu(candy::Model * gpu_ptr, void * grid_vector_data_ptr, std::uintptr_t stream_ptr) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

// Copy data from GPU to CPU
void * candy::Model::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// Check if these is a negative parameter in the model
bool candy::Model::check_negative(void) const noexcept {
    return std::all_of(this->parameters_.cbegin(), this->parameters_.cend(),
                       [](const double & value) { return value >= 0; });
}

// Check if a given data shape is compatible with the current model
bool candy::Model::check_compatible_shape(const Index & shape) const noexcept {
    std::pair<Index::const_iterator, Index::const_iterator> iter = std::mismatch(
        this->rshape_.cbegin(), this->rshape_.cend(), shape.cbegin(),
        [this](const std::uint64_t & rshape, const std::uint64_t & shape) {
            return rshape == shape * this->rank_;
        });
    return iter.first == this->rshape_.end();
}

// Write model into a file
void candy::Model::save(const std::string & fname, bool lock) const {
    // open file
    std::FILE * file_stream = std::fopen(fname.c_str(), "wb");
    if (file_stream == nullptr) {
        Fatal<std::filesystem::filesystem_error>("Cannot create file %s\n", fname.c_str());
    }
    FileLock flock(file_stream);
    // lambda write file
    auto write_lambda = [&file_stream](const void * data, std::size_t elem_size, std::size_t n_elems) {
        std::size_t success_written = std::fwrite(data, elem_size, n_elems, file_stream);
        if (success_written < n_elems) {
            Fatal<std::filesystem::filesystem_error>("Error occurred when writing the file.\n");
        }
    };
    if (lock) {
        flock.lock();
    }
    // write ndim and size
    std::uint64_t meta_data[3] = {this->rank_, this->ndim_, this->num_params()};
    write_lambda(meta_data, sizeof(std::uint64_t), 3);
    // write rshape
    write_lambda(this->rshape_.data(), sizeof(std::uint64_t), this->ndim_);
    // write parameters
    write_lambda(this->parameters_.data(), sizeof(double), this->num_params());
    // close file
    if (lock) {
        flock.unlock();
    }
    std::fclose(file_stream);
}

// Read polynomial data from a file
void candy::Model::load(const std::string & fname, bool lock) {
    // open file
    std::FILE * file_stream = std::fopen(fname.c_str(), "rb");
    if (file_stream == nullptr) {
        Fatal<std::filesystem::filesystem_error>("Cannot read file %s\n", fname.c_str());
    }
    FileLock flock(file_stream);
    // lambda read file
    auto read_lambda = [&file_stream](void * data, std::size_t elem_size, std::size_t n_elems) {
        std::size_t success_read = std::fread(data, elem_size, n_elems, file_stream);
        if (success_read < n_elems) {
            Fatal<std::filesystem::filesystem_error>("Error occurred when reading the file.\n");
        }
    };
    if (lock) {
        flock.lock_shared();
    }
    // read ndim and size
    std::uint64_t meta_data[3];
    read_lambda(meta_data, sizeof(std::uint64_t), 3);
    this->rank_ = meta_data[0];
    // read rshape
    this->ndim_ = meta_data[1];
    this->rshape_.fill(0);
    read_lambda(this->rshape_.data(), sizeof(std::uint64_t), this->ndim_);
    // read parameters
    this->parameters_ = DoubleVec(meta_data[2]);
    read_lambda(this->parameters_.data(), sizeof(double), meta_data[2]);
    // calculate pointers
    this->param_vectors_.fill(nullptr);
    ptr_to_subsequence(this->parameters_.data(), this->rshape_.data(), this->ndim_, this->param_vectors_.data());
    // close file
    if (lock) {
        flock.unlock();
    }
    std::fclose(file_stream);
}

// String representation
std::string candy::Model::str(void) const {
    std::ostringstream out_stream;
    out_stream << "<Model(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
        out_stream << ((i_dim != 0) ? " " : "");
        out_stream << "<";
        std::uint64_t dim_shape = this->rshape_[i_dim] / this->rank_;
        for (std::uint64_t i_index = 0; i_index < dim_shape; i_index++) {
            DoubleVec rank_vector;
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
