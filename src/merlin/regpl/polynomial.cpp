// Copyright 2024 quocdang1998
#include "merlin/regpl/polynomial.hpp"

#include <algorithm>     // std::stable_sort
#include <cinttypes>     // PRIu64
#include <mutex>         // std::unique_lock
#include <numeric>       // std::iota
#include <shared_mutex>  // std::shared_lock
#include <sstream>       // std::ostringstream

#include "merlin/io/file_lock.hpp"     // merlin::io::FileLock
#include "merlin/io/file_pointer.hpp"  // merlin::io::FilePointer, merlin::io::open_file
#include "merlin/io/io_engine.hpp"     // merlin::io::ReadEngine, merlin::io::WriteEngine
#include "merlin/logger.hpp"           // merlin::Fatal, merlin::cuda_compile_error
#include "merlin/memory.hpp"           // merlin::memcpy_cpu_to_gpu, merlin::memcpy_gpu_to_cpu
#include "merlin/utils.hpp"            // merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------------------------------------------------

// Constructor of an empty polynomial from order per dimension
regpl::Polynomial::Polynomial(const Index & order) : order_(order) {
    this->coeff_ = DoubleVec(prod_elements(order.data(), order.size()));
}

// Constructor of a pre-allocated array of coefficients and order per dimension
regpl::Polynomial::Polynomial(const DoubleVec & coeff, const Index & order) : coeff_(coeff), order_(order) {
    if (coeff.size() != prod_elements(order.data(), order.size())) {
        Fatal<std::invalid_argument>("Insufficient number of coefficients provided for the given order_per_dim.\n");
    }
}

// Set coefficients in case of a sparse polynomial
void regpl::Polynomial::set(double * coeff, const UIntVec & term_index) {
    // check argument
    UIntVec sorted_idx(term_index.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    auto sort_lambda = [&term_index](const std::uint64_t & i1, const std::uint64_t & i2) {
        return term_index[i1] < term_index[i2];
    };
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(), sort_lambda);
    for (std::uint64_t i_term = 1; i_term < term_index.size(); i_term++) {
        if (term_index[sorted_idx[i_term]] == term_index[sorted_idx[i_term - 1]]) {
            Fatal<std::invalid_argument>("Found duplicated index.\n");
        }
        if (term_index[i_term] >= this->size()) {
            Fatal<std::invalid_argument>("Invalid index (index bigger than number of coefficients).\n");
        }
    }
    // add to coeff
    for (std::uint64_t i_term = 0; i_term < term_index.size(); i_term++) {
        this->coeff_[term_index[i_term]] = coeff[i_term];
    }
}

// Copy data to a pre-allocated memory
void * regpl::Polynomial::copy_to_gpu(regpl::Polynomial * gpu_ptr, void * coeff_data_ptr,
                                      std::uintptr_t stream_ptr) const {
    // copy data of coefficient vector
    memcpy_cpu_to_gpu(coeff_data_ptr, this->coeff_.data(), this->size() * sizeof(double), stream_ptr);
    // initialize buffer to store data of the copy before cloning it to GPU
    regpl::Polynomial copy_on_gpu;
    // shallow copy of coefficients, orders and term index
    double * coeff_ptr = reinterpret_cast<double *>(coeff_data_ptr);
    copy_on_gpu.coeff_.assign(coeff_ptr, this->size());
    copy_on_gpu.order_ = this->order_;
    memcpy_cpu_to_gpu(gpu_ptr, &copy_on_gpu, sizeof(regpl::Polynomial), stream_ptr);
    return reinterpret_cast<void *>(coeff_ptr + this->size());
}

// Copy data from GPU to CPU
void * regpl::Polynomial::copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr) noexcept {
    memcpy_gpu_to_cpu(this->coeff_.data(), data_from_gpu, this->size() * sizeof(double), stream_ptr);
    return reinterpret_cast<void *>(data_from_gpu + this->size());
}

// Write polynomial data into a file
void regpl::Polynomial::save(const std::string & fname, std::uint64_t offset, bool lock) const {
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
    // write shape
    std::uint64_t ndim = this->ndim();
    uint_write.write(&ndim, 1);
    uint_write.write(this->order_.data(), this->order_.size());
    // write coefficients
    float_write.write(this->coeff_.data(), this->coeff_.size());
}

// Read polynomial data from a file
void regpl::Polynomial::load(const std::string & fname, std::uint64_t offset, bool lock) {
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
    // read shape
    std::uint64_t ndim;
    uint_read.read(&ndim, 1);
    this->order_.resize(ndim);
    uint_read.read(this->order_.data(), this->order_.size());
    // read coefficients
    std::uint64_t size = prod_elements(this->order_.data(), this->order_.size());
    this->coeff_ = DoubleVec(size);
    float_read.read(this->coeff_.data(), this->coeff_.size());
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
