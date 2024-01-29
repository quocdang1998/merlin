// Copyright 2024 quocdang1998
#include "merlin/regpl/polynomial.hpp"

#include <algorithm>  // std::iota, std::stable_sort, std::unique
#include <cinttypes>  // PRIu64
#include <cmath>    // std::pow
#include <cstdio>  // std::FILE, std::fread, std::fwrite, std::fopen, std::fclose
#include <filesystem>  // std::filesystem::filesystem_error
#include <sstream>  // std::ostringstream

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/utils.hpp"  // merlin::prod_elements, merlin::ndim_to_contiguous_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// read data from file
static void read_data(void * dest, std::uint64_t size, std::uint64_t count, std::FILE * fstream) {
    if (std::fread(dest, size, count, fstream) != count) {
        FAILURE(std::runtime_error, "Failed to read data from file.\n");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------------------------------------------------

// Constructor of an empty polynomial from order per dimension
regpl::Polynomial::Polynomial(const intvec & order_per_dim) :
order_(order_per_dim), coeff_(prod_elements(order_per_dim)), term_idx_(this->coeff_.size()), full_n_(coeff_.size()) {
    std::iota(this->term_idx_.begin(), this->term_idx_.end(), 0);
}

// Constructor of a pre-allocated array of coefficients and order per dimension
regpl::Polynomial::Polynomial(const floatvec & coeff_data, const intvec & order_per_dim) :
order_(order_per_dim), coeff_(coeff_data), term_idx_(coeff_data.size()), full_n_(coeff_data.size()) {
    if (coeff_data.size() != prod_elements(order_per_dim)) {
        FAILURE(std::invalid_argument, "Insufficient number of coefficients provided for the given order_per_dim.\n");
    }
    std::iota(this->term_idx_.begin(), this->term_idx_.end(), 0);
}

// Constructor of a sparse polynomial
regpl::Polynomial::Polynomial(const floatvec & coeff_data, const intvec & order_per_dim, const intvec & term_index) :
order_(order_per_dim), coeff_(coeff_data.size()), term_idx_(coeff_data.size()), full_n_(prod_elements(order_per_dim)) {
    // check argument
    if (term_index.size() % order_per_dim.size() != 0) {
        FAILURE(std::invalid_argument, "Size of argument term_index must divide size of order_per_dim.\n");
    }
    if (coeff_data.size() * order_per_dim.size() != term_index.size()) {
        FAILURE(std::invalid_argument, "Inconsistent argument provided.\n");
    }
    // initialize sort index
    intvec sorted_idx(coeff_data.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    // get term index
    for (std::uint64_t i_term = 0; i_term < coeff_data.size(); i_term++) {
        const std::uint64_t * term_idx_ptr = term_index.data() + i_term * order_per_dim.size();
        // check for consistent
        for (std::uint64_t i = 0; i < order_per_dim.size(); i++) {
            if (term_idx_ptr[i] >= order_per_dim[i]) {
                FAILURE(std::invalid_argument, "Invalid input at term %" PRIu64 ".\n", i_term);
            }
        }
        intvec term_idx;
        term_idx.assign(const_cast<std::uint64_t *>(term_idx_ptr), order_per_dim.size());
        this->term_idx_[i_term] = ndim_to_contiguous_idx(term_idx, order_per_dim);
    }
    // sort term index
    auto sort_lambda = [this](const std::uint64_t & i1, const std::uint64_t & i2) {
        return this->term_idx_[i1] < this->term_idx_[i2];
    };
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(), sort_lambda);
    intvec argument_idx(this->term_idx_);
    for (std::uint64_t i_term = 0; i_term < coeff_data.size(); i_term++) {
        this->term_idx_[i_term] = argument_idx[sorted_idx[i_term]];
        this->coeff_[i_term] = coeff_data[sorted_idx[i_term]];
    }
    if (std::unique(this->term_idx_.begin(), this->term_idx_.end()) != this->term_idx_.end()) {
        FAILURE(std::invalid_argument, "Duplicate index detected.\n");
    }
}

// Generate Vandermonde matrix
array::Array regpl::Polynomial::calc_vandermonde(const array::Array & grid_points, std::uint64_t n_threads) const {
    // check argument
    if (grid_points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of 2 dimensions.\n");
    }
    if (grid_points.shape()[1] != this->ndim()) {
        FAILURE(std::invalid_argument, "Points in grid have a different number of dimension as the polynomial.\n");
    }
    // allocate matrix
    std::uint64_t num_points = grid_points.shape()[0];
    array::Array vandermonde_matrix({num_points, this->size()});
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        intvec point_index(2);
        // loop for each point in the grid
        for (std::uint64_t i_point = thread_idx; i_point < num_points; i_point += n_threads) {
            // get pointer to corresponding row
            double * row_data = vandermonde_matrix.data() + i_point * this->size();
            point_index[0] = i_point;
            // loop for each monomial
            for (std::uint64_t i_term = 0; i_term < this->size(); i_term++) {
                row_data[i_term] = 1;
                // calculate monomial expansion for each dimension
                std::uint64_t cum_prod = 1;
                for (std::int64_t i_dim = this->ndim() - 1; i_dim >= 0; i_dim--) {
                    std::uint64_t nd_index = (this->term_idx_[i_term] / cum_prod) % this->order_[i_dim];
                    point_index[1] = i_dim;
                    row_data[i_term] *= std::pow(grid_points.get(point_index), nd_index);
                    cum_prod *= this->order_[i_dim];
                }
            }
        }
    }
    return vandermonde_matrix;
}

#ifndef __MERLIN_CUDA__

// Copy data to a pre-allocated memory
void * regpl::Polynomial::copy_to_gpu(regpl::Polynomial * gpu_ptr, void * coeff_data_ptr,
                                      std::uintptr_t stream_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

// Copy data from GPU to CPU
void * regpl::Polynomial::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to use this method.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

// Write polynomial data into a file
void regpl::Polynomial::serialize(const std::string & fname) const {
    // open file
    std::FILE * file_stream = std::fopen(fname.c_str(), "wb");
    if (file_stream == nullptr) {
        FAILURE(std::filesystem::filesystem_error, "Cannot create file %s\n", fname.c_str());
    }
    // write ndim and size
    std::uint64_t meta_data[2] = {this->ndim(), this->size()};
    std::fwrite(meta_data, sizeof(std::uint64_t), 2, file_stream);
    // write order per dim
    std::fwrite(this->order_.data(), sizeof(std::uint64_t), this->ndim(), file_stream);
    // write coefficients and coeff index
    std::fwrite(this->coeff_.data(), sizeof(double), this->size(), file_stream);
    std::fwrite(this->term_idx_.data(), sizeof(std::uint64_t), this->size(), file_stream);
    // close file
    std::fclose(file_stream);
}

// Read polynomial data from a file
void regpl::Polynomial::deserialize(const std::string & fname) {
    // open file
    std::FILE * file_stream = std::fopen(fname.c_str(), "rb");
    if (file_stream == nullptr) {
        FAILURE(std::filesystem::filesystem_error, "Cannot read file %s\n", fname.c_str());
    }
    // read ndim and size
    std::uint64_t meta_data[2];
    read_data(meta_data, sizeof(std::uint64_t), 2, file_stream);
    // read order per dim
    this->order_ = intvec(meta_data[0]);
    read_data(this->order_.data(), sizeof(std::uint64_t), meta_data[0], file_stream);
    // read coefficients and coeff index
    this->coeff_ = floatvec(meta_data[1]);
    read_data(this->coeff_.data(), sizeof(double), meta_data[1], file_stream);
    this->term_idx_ = intvec(meta_data[1]);
    read_data(this->term_idx_.data(), sizeof(std::uint64_t), meta_data[1], file_stream);
    // close file
    std::fclose(file_stream);
}

// String representation
std::string regpl::Polynomial::str(void) const {
    std::ostringstream os;
    os << "<Polynomial(";
    os << "order=" << this->order_.str() << ", ";
    os << "coeff=" << this->coeff_.str() << ", ";
    os << "term_index=" << this->term_idx_.str();
    os << ")";
    return os.str();
}

}  // namespace merlin
