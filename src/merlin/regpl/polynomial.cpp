// Copyright 2024 quocdang1998
#include "merlin/regpl/polynomial.hpp"

#include <algorithm>  // std::stable_sort
#include <cinttypes>  // PRIu64
#include <filesystem>  // std::filesystem::filesystem_error
#include <numeric>  // std::iota
#include <sstream>  // std::ostringstream

#include "merlin/filelock.hpp"  // merlin::FileLock
#include "merlin/logger.hpp"  // FAILURE, merlin::cuda_compile_error
#include "merlin/utils.hpp"  // merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------------------------------------------------

// Constructor of an empty polynomial from order per dimension
regpl::Polynomial::Polynomial(const intvec & order) : order_(order), coeff_(prod_elements(order)) {}

// Constructor of a pre-allocated array of coefficients and order per dimension
regpl::Polynomial::Polynomial(const floatvec & coeff, const intvec & order) :
coeff_(coeff), order_(order) {
    if (coeff.size() != prod_elements(order)) {
        FAILURE(std::invalid_argument, "Insufficient number of coefficients provided for the given order_per_dim.\n");
    }
}

// Constructor of a sparse polynomial
regpl::Polynomial::Polynomial(const floatvec & coeff, const intvec & order, const intvec & term_index) :
order_(order), coeff_(prod_elements(order)) {
    // check argument
    intvec sorted_idx(term_index.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    auto sort_lambda = [&term_index](const std::uint64_t & i1, const std::uint64_t & i2) {
        return term_index[i1] < term_index[i2];
    };
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(), sort_lambda);
    for (std::uint64_t i_term = 1; i_term < term_index.size(); i_term++) {
        if (term_index[sorted_idx[i_term]] == term_index[sorted_idx[i_term-1]]) {
            FAILURE(std::invalid_argument, "Found duplicated index.\n");
        }
    }
    if (coeff.size() != term_index.size()) {
        FAILURE(std::invalid_argument, "Inconsistent number of coefficients and index array.\n");
    }
    // add to coeff
    for (std::uint64_t i_term = 0; i_term < term_index.size(); i_term++) {
        this->coeff_[term_index[i_term]] = coeff[i_term];
    }
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
void regpl::Polynomial::save(const std::string & fname, bool lock) const {
    // open file
    std::FILE * file_stream = std::fopen(fname.c_str(), "wb");
    if (file_stream == nullptr) {
        FAILURE(std::filesystem::filesystem_error, "Cannot create file %s\n", fname.c_str());
    }
    FileLock flock(file_stream);
    // lambda write file
    auto write_lambda = [&file_stream] (const void * data, std::size_t elem_size, std::size_t n_elems) {
        std::size_t success_written = std::fwrite(data, elem_size, n_elems, file_stream);
        if (success_written < n_elems) {
            FAILURE(std::filesystem::filesystem_error, "Error occurred when writing the file.\n");
        }
    };
    if (lock) {
        flock.lock();
    }
    // write ndim and size
    std::uint64_t meta_data[2] = {this->ndim(), this->size()};
    write_lambda(meta_data, sizeof(std::uint64_t), 2);
    // write order per dim
    write_lambda(this->order_.data(), sizeof(std::uint64_t), this->ndim());
    // write coefficients and coeff index
    write_lambda(this->coeff_.data(), sizeof(double), this->size());
    // close file
    if (lock) {
        flock.unlock();
    }
    std::fclose(file_stream);
}

// Read polynomial data from a file
void regpl::Polynomial::load(const std::string & fname, bool lock) {
    // open file
    std::FILE * file_stream = std::fopen(fname.c_str(), "rb");
    if (file_stream == nullptr) {
        FAILURE(std::filesystem::filesystem_error, "Cannot read file %s\n", fname.c_str());
    }
    FileLock flock(file_stream);
    // lambda read file
    auto read_lambda = [&file_stream] (void * data, std::size_t elem_size, std::size_t n_elems) {
        std::size_t success_read = std::fread(data, elem_size, n_elems, file_stream);
        if (success_read < n_elems) {
            FAILURE(std::filesystem::filesystem_error, "Error occurred when reading the file.\n");
        }
    };
    if (lock) {
        flock.lock_shared();
    }
    // read ndim and size
    std::uint64_t meta_data[2];
    read_lambda(meta_data, sizeof(std::uint64_t), 2);
    // read order per dim
    this->order_ = intvec(meta_data[0]);
    read_lambda(this->order_.data(), sizeof(std::uint64_t), meta_data[0]);
    // read coefficients and coeff index
    this->coeff_ = floatvec(meta_data[1]);
    read_lambda(this->coeff_.data(), sizeof(double), meta_data[1]);
    // close file
    if (lock) {
        flock.unlock();
    }
    std::fclose(file_stream);
}

// String representation
std::string regpl::Polynomial::str(void) const {
    std::ostringstream os;
    os << "<Polynomial(";
    os << "order=" << this->order_.str() << ", ";
    os << "coeff=" << this->coeff_.str() << ")>";
    return os.str();
}

}  // namespace merlin
