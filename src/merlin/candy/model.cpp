// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <algorithm>     // std::copy_n
#include <cmath>         // std::pow, std::sqrt
#include <cstdio>        // std::fopen, std::fseek
#include <fstream>       // std::ifstream, std::ofstream
#include <iterator>      // std::distance
#include <memory>        // std::unique_ptr
#include <mutex>         // std::unique_lock
#include <shared_mutex>  // std::shared_lock
#include <sstream>       // std::ostringstream
#include <type_traits>   // std::add_pointer

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/env.hpp"              // merlin::Environment
#include "merlin/io/byteswap.hpp"      // merlin::io::little_endian
#include "merlin/io/file_lock.hpp"     // merlin::io::FileLock
#include "merlin/io/file_pointer.hpp"  // merlin::io::FilePointer, merlin::io::create_file
#include "merlin/io/io_engine.hpp"     // merlin::io::ReadEngine, merlin::io::WriteEngine
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/memory.hpp"           // merlin::memcpy_cpu_to_gpu, merlin::memcpy_gpu_to_cpu
#include "merlin/slice.hpp"            // merlin::SliceArray
#include "merlin/utils.hpp"            // merlin::ptr_to_subsequence

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Calculate rstride and offset
static std::uint64_t get_rstride(const Index & shape, Index & offset) {
    offset.resize(shape.size());
    std::uint64_t cum_sum = 0;
    for (std::uint64_t i_dim = 0; i_dim < shape.size(); i_dim++) {
        offset[i_dim] = cum_sum;
        cum_sum += shape[i_dim];
    }
    return cum_sum;
}

// ---------------------------------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------------------------------

// Calculate offset per dimension
void assign_offset_rstride(void);

// Constructor from shape and rank
candy::Model::Model(const Index & shape, std::uint64_t rank) : shape_(shape), rank_(rank) {
    // check argument
    if (rank == 0) {
        Fatal<std::invalid_argument>("Cannot initialize a canonical model with rank zero.\n");
    }
    // get r-stride and offset
    this->rstride_ = get_rstride(shape, this->offset_);
    // reserve parameter vector
    this->parameters_ = DoubleVec(this->rstride_ * rank);
}

// Constructor from model values
candy::Model::Model(const DVecArray & param_vectors, std::uint64_t rank) : rank_(rank) {
    // check argument
    if (rank == 0) {
        Fatal<std::invalid_argument>("Cannot initialize a canonical model with rank zero.\n");
    }
    // get and check shape
    std::uint64_t ndim = param_vectors.size();
    this->shape_.resize(ndim);
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (param_vectors[i_dim].size() % rank != 0) {
            Fatal<std::invalid_argument>("Size of concatenated parameter vectors must be divisible by the rank.\n");
        }
        this->shape_[i_dim] = param_vectors[i_dim].size() / rank;
    }
    // get r-stride and offset
    this->rstride_ = get_rstride(this->shape_, this->offset_);
    // copy to parameter vector
    this->parameters_ = DoubleVec(this->rstride_ * rank);
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        const DoubleVec & param_vector = param_vectors[i_dim];
        for (std::uint64_t r = 0; r < rank; r++) {
            std::copy_n(param_vector.data() + r * this->shape_[i_dim], this->shape_[i_dim],
                        this->parameters_.data() + r * this->rstride_ + this->offset_[i_dim]);
        }
    }
}

// Initialize values of model based on train data
void candy::Model::initialize(const array::Array & train_data, candy::Randomizer * randomizer) {
    // check shape
    if (this->shape_ != train_data.shape()) {
        Fatal<std::invalid_argument>("Incompatible shape between data and model.\n");
    }
    // initialize model parameters for each dimension
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        SliceArray slice_division(this->ndim());
        std::uint64_t num_division = this->shape_[i_dim];
        // loop on each hyper-slice
        for (std::uint64_t i_division = 0; i_division < num_division; i_division++) {
            // calculate mean value for each hyper-slice
            slice_division[i_dim] = Slice(i_division, i_division + 1, 1);
            array::Array sub_data = train_data.get_sub_array(slice_division);
            auto [mean, variance] = sub_data.get_mean_variance();
            double stdeviation = std::sqrt(variance);
            // calculate mean <- (mean/rank)^(1/ndim) and stdeviation <- stdeviation * new_mean / old_mean
            if (mean != 0.0) {
                stdeviation /= mean;
                mean = std::pow(mean / this->rank_, 1.f / this->ndim());
                stdeviation *= mean;
            }
            // initializer randomizer
            candy::set_params(randomizer[i_dim], mean, stdeviation);
            // initialize model parameters
            for (std::uint64_t r = 0; r < this->rank_; r++) {
                this->get(i_dim, i_division, r) = candy::sample(randomizer[i_dim]);
            }
        }
    }
}

// Copy data to a pre-allocated memory
void * candy::Model::copy_to_gpu(candy::Model * gpu_ptr, void * parameters_data_ptr, std::uintptr_t stream_ptr) const {
    // copy parameter data to GPU
    memcpy_cpu_to_gpu(parameters_data_ptr, this->parameters_.data(), this->num_params() * sizeof(double), stream_ptr);
    // initialize cloned version on GPU
    candy::Model cloned_obj;
    double * parameters_data = reinterpret_cast<double *>(parameters_data_ptr);
    cloned_obj.parameters_.assign(parameters_data, this->num_params());
    cloned_obj.shape_ = this->shape_;
    cloned_obj.rank_ = this->rank_;
    cloned_obj.rstride_ = this->rstride_;
    cloned_obj.offset_ = this->offset_;
    memcpy_cpu_to_gpu(gpu_ptr, &cloned_obj, sizeof(candy::Model), stream_ptr);
    return reinterpret_cast<void *>(parameters_data + this->num_params());
}

// Copy data from GPU to CPU
void * candy::Model::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    memcpy_gpu_to_cpu(this->parameters_.data(), data_from_gpu, this->num_params() * sizeof(double), stream_ptr);
    return reinterpret_cast<void *>(data_from_gpu + this->num_params());
}

// Write model into a file
void candy::Model::save(const std::string & fname, std::uint64_t offset, bool lock) const {
    // open file
    io::FilePointer file_stream = io::open_file(fname.c_str());
    file_stream.seek(offset);
    // acquire file lock
    io::FileLock flock(file_stream.get());
    std::unique_lock<io::FileLock> guard = ((lock) ? std::unique_lock<io::FileLock>(flock)
                                                   : std::unique_lock<io::FileLock>());
    // initialize write engines
    io::WriteEngine<std::uint64_t> uint_write(file_stream.get());
    io::WriteEngine<double> float_write(file_stream.get());
    // write rank and ndim
    std::uint64_t meta_data[2] = {this->rank_, this->ndim()};
    uint_write.write(meta_data, 2);
    // write shape
    uint_write.write(this->shape_.data(), this->shape_.size());
    // write parameters
    float_write.write(this->parameters_.data(), this->parameters_.size());
}

// Read polynomial data from a file
void candy::Model::load(const std::string & fname, std::uint64_t offset, bool lock) {
    // open file
    io::FilePointer file_stream = io::open_file(fname.c_str());
    file_stream.seek(offset);
    // acquire file lock
    io::FileLock flock(file_stream.get());
    std::shared_lock<io::FileLock> guard = ((lock) ? std::shared_lock<io::FileLock>(flock)
                                                   : std::shared_lock<io::FileLock>());
    // initialize read engines
    io::ReadEngine<std::uint64_t> uint_read(file_stream.get());
    io::ReadEngine<double> float_read(file_stream.get());
    // read rank, ndim and shape
    std::uint64_t meta_data[2];
    uint_read.read(meta_data, 2);
    this->rank_ = meta_data[0];
    // read shape and calculate offset
    this->shape_.resize(meta_data[1]);
    uint_read.read(this->shape_.data(), this->shape_.size());
    this->rstride_ = get_rstride(this->shape_, this->offset_);
    // read parameters
    this->parameters_ = DoubleVec(this->rstride_ * this->rank_);
    float_read.read(this->parameters_.data(), this->parameters_.size());
}

// String representation
std::string candy::Model::str(void) const {
    std::ostringstream out_stream;
    out_stream << "<Model(";
    out_stream << "rank=" << this->rank_ << ", ";
    out_stream << "shape=" << this->shape_.str() << ", ";
    out_stream << "parameters=<";
    for (std::uint64_t r = 0; r < this->rank_; r++) {
        if (r != 0) {
            out_stream << " ";
        }
        out_stream << "<";
        for (std::uint64_t i = 0; i < this->ndim(); i++) {
            DoubleView dim_view(this->parameters_.data() + r * this->rstride_ + this->offset_[i], this->shape_[i]);
            out_stream << dim_view.str();
        }
        out_stream << ">";
    }
    out_stream << ">";
    out_stream << ")>";
    return out_stream.str();
}

}  // namespace merlin
