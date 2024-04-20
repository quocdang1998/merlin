// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <cinttypes>     // PRIu64
#include <cstring>       // std::memcpy
#include <filesystem>    // std::filesystem::filesystem_error, std::filesystem::file_size, std::filesystem::resize_file
#include <functional>    // std::bind, std::placeholders
#include <ios>           // std::ios_base::failure
#include <mutex>         // std::unique_lock
#include <shared_mutex>  // std::shared_lock

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/config.hpp"           // merlin::max_dim
#include "merlin/logger.hpp"           // merlin::Fatal
#include "merlin/platform.hpp"         // __MERLIN_LINUX__, __MERLIN_WINDOWS__
#include "merlin/utils.hpp"            // merlin::flip_endianess, merlin::flip_range

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Data read/write
// ---------------------------------------------------------------------------------------------------------------------

// Read an array from file
static inline void read_from_file(void * dest, std::FILE * file, const void * src, std::uint64_t bytes,
                                  bool same_endianess) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (std::fread(dest, sizeof(double), count, file) != count) {
        Fatal<std::ios_base::failure>("Read file error.\n");
    }
    if (!same_endianess) {
        flip_range(reinterpret_cast<std::uint64_t *>(dest), count);
    }
}

// Write an array from file
static inline void write_to_file(std::FILE * file, void * dest, const void * src, std::uint64_t bytes,
                                 bool same_endianess) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(dest), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (same_endianess) {
        if (std::fwrite(src, sizeof(double), bytes / sizeof(double), file) != count) {
            Fatal<std::ios_base::failure>("Write file error.\n");
        }
    } else {
        char * flipped_src = new char[bytes];
        std::memcpy(flipped_src, src, bytes);
        flip_range(reinterpret_cast<std::uint64_t *>(flipped_src), count);
        if (std::fwrite(flipped_src, sizeof(double), bytes / sizeof(double), file) != count) {
            Fatal<std::ios_base::failure>("Write file error.\n");
        }
        delete[] flipped_src;
    }
}

// Check if file exists
static inline bool check_file_exist(const char * name) noexcept {
    if (std::FILE * file_ptr = std::fopen(name, "r")) {
        std::fclose(file_ptr);
        return true;
    } else {
        return false;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Stock
// ---------------------------------------------------------------------------------------------------------------------

// Read metadata from file
std::uint64_t array::Stock::read_metadata(void) {
    // initialize lock guard
    std::shared_lock<FileLock> lock = ((this->thread_safe_) ? std::shared_lock<FileLock>(this->flock_)
                                                            : std::shared_lock<FileLock>());
    // zero fill shape and stride
    this->shape_.fill(0);
    this->strides_.fill(0);
    // read ndim and detect endianness
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    if (std::fread(&(this->ndim_), sizeof(std::uint64_t), 1, this->file_ptr_) != 1) {
        Fatal<std::ios_base::failure>("Read file error.\n");
    }
    if (this->ndim_ > max_dim) {
        this->ndim_ = flip_endianess(this->ndim_);
        if (this->ndim_ > max_dim) {
            Fatal<std::invalid_argument>("Ndim of the stock file is bigger than max_dim.\n");
        } else {
            this->same_endianess_ = false;
        }
    }
    // read data shape
    if (std::fread(this->shape_.data(), sizeof(std::uint64_t), this->ndim_, this->file_ptr_) != this->ndim_) {
        Fatal<std::ios_base::failure>("Read file error.\n");
    }
    if (!this->same_endianess_) {
        flip_range(this->shape_.data(), this->ndim_);
    }
    std::uint64_t cursor = std::ftell(this->file_ptr_);
    // calculate stride and assign data pointer
    this->calc_array_size();
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    this->data_ = reinterpret_cast<double *>(cursor);
    // check file size
    std::uint64_t file_size = std::filesystem::file_size(this->filename_);
    std::uint64_t expected_size = this->offset_ + (1 + this->ndim_) * sizeof(std::uint64_t);
    expected_size += this->size() * sizeof(double);
    if (file_size < expected_size) {
        Fatal<std::filesystem::filesystem_error>("Expected filesize of at least %" PRIu64 ", got %" PRIu64 ".\n",
                                                 expected_size, file_size);
    }
    return cursor;
}

// Write metadata to file at offset position
std::uint64_t array::Stock::write_metadata(void) {
    // acquire file lock
    std::unique_lock<FileLock> lock = ((this->thread_safe_) ? std::unique_lock<FileLock>(this->flock_)
                                                            : std::unique_lock<FileLock>());
    // write ndim and shape data to file at position offset
    std::uint64_t ndim = this->ndim();
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    if (std::fwrite(&ndim, sizeof(std::uint64_t), 1, this->file_ptr_) != 1) {
        Fatal<std::ios_base::failure>("Write file error.\n");
    }
    if (std::fwrite(this->shape_.data(), sizeof(std::uint64_t), ndim, this->file_ptr_) != ndim) {
        Fatal<std::ios_base::failure>("Write file error.\n");
    }
    std::uint64_t cursor = std::ftell(this->file_ptr_);
    // change data pointer to current cursor
    this->data_ = reinterpret_cast<double *>(cursor);
    return cursor;
}

// Construct empty Array from shape vector
array::Stock::Stock(const std::string & filename, const Index & shape, std::uint64_t offset, bool thread_safe) :
array::NdData(shape), filename_(filename), offset_(0), thread_safe_(thread_safe) {
    // create file if not exists
    bool file_exist = check_file_exist(filename.c_str());
    if (!file_exist) {
        std::FILE * temporary = std::fopen(filename.c_str(), "wb");
        std::fclose(temporary);
    }
    // resize file
    std::uint64_t file_size = std::filesystem::file_size(filename);
    std::uint64_t new_file_size = (1 + this->ndim()) * sizeof(std::uint64_t) + this->size() * sizeof(double);
    if (offset + new_file_size > file_size) {
        if (offset < file_size) {
            new_file_size += file_size - offset;
        } else {
            new_file_size += offset;
        }
        std::filesystem::resize_file(filename, new_file_size);
    }
    // assign file pointer
    this->file_ptr_ = std::fopen(filename.c_str(), "rb+");
    this->flock_ = FileLock(this->file_ptr_);
    // write metatdata
    this->write_metadata();
    this->release = true;
}

// Constructor from filename
array::Stock::Stock(const std::string & filename, std::uint64_t offset, bool thread_safe) :
filename_(filename), offset_(offset), thread_safe_(thread_safe) {
    // check if file exists
    bool file_exist = check_file_exist(filename.c_str());
    if (!file_exist) {
        Fatal<std::filesystem::filesystem_error>("Cannot find file \"%s\", please make sure that the file exists.\n",
                                                 filename.c_str());
    }
    // open file
    this->file_ptr_ = std::fopen(filename.c_str(), "rb+");
    this->flock_ = FileLock(this->file_ptr_);
    this->read_metadata();
    this->release = true;
}

// Get value of element at a n-dim index
double array::Stock::get(const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::shared_lock<FileLock> lock = ((this->thread_safe_) ? std::shared_lock<FileLock>(this->flock_)
                                                            : std::shared_lock<FileLock>());
    double result;
    read_from_file(&result, this->file_ptr_, reinterpret_cast<double *>(data_ptr), sizeof(double),
                   this->same_endianess_);
    return result;
}

// Get value of element at a C-contiguous index
double array::Stock::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::shared_lock<FileLock> lock = ((this->thread_safe_) ? std::shared_lock<FileLock>(this->flock_)
                                                            : std::shared_lock<FileLock>());
    double result;
    read_from_file(&result, this->file_ptr_, reinterpret_cast<double *>(data_ptr), sizeof(double),
                   this->same_endianess_);
    return result;
}

// Set value of element at a n-dim index
void array::Stock::set(const Index & index, double value) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::unique_lock<FileLock> lock = ((this->thread_safe_) ? std::unique_lock<FileLock>(this->flock_)
                                                            : std::unique_lock<FileLock>());
    write_to_file(this->file_ptr_, reinterpret_cast<double *>(data_ptr), &value, sizeof(double), this->same_endianess_);
}

// Set value of element at a C-contiguous index
void array::Stock::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::unique_lock<FileLock> lock = ((this->thread_safe_) ? std::unique_lock<FileLock>(this->flock_)
                                                            : std::unique_lock<FileLock>());
    write_to_file(this->file_ptr_, reinterpret_cast<double *>(data_ptr), &value, sizeof(double), this->same_endianess_);
}

// Set value of all elements
void array::Stock::fill(double value) {
    auto write_func = std::bind(write_to_file, this->file_ptr_, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3, this->same_endianess_);
    std::unique_lock<FileLock> lock = ((this->thread_safe_) ? std::unique_lock<FileLock>(this->flock_)
                                                            : std::unique_lock<FileLock>());
    array::fill(this, value, write_func);
}

// Calculate mean and variance of all non-zero and finite elements
std::array<double, 2> array::Stock::get_mean_variance(void) const {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, this->file_ptr_, std::placeholders::_2,
                               std::placeholders::_3, this->same_endianess_);
    std::shared_lock<FileLock> lock = ((this->thread_safe_) ? std::shared_lock<FileLock>(this->flock_)
                                                            : std::shared_lock<FileLock>());
    return array::stat(this, read_func);
}

// Write data from an array to a file
void array::Stock::record_data_to_file(const array::Array & src) {
    auto write_func = std::bind(write_to_file, this->file_ptr_, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3, this->same_endianess_);
    std::unique_lock<FileLock> lock = ((this->thread_safe_) ? std::unique_lock<FileLock>(this->flock_)
                                                            : std::unique_lock<FileLock>());
    copy(this, &src, write_func);
}

// String representation
std::string array::Stock::str(bool first_call) const { return array::print(this, "Stock", first_call); }

// Destructor
array::Stock::~Stock(void) {
    if (this->release && (this->file_ptr_ != nullptr)) {
        int err_ = std::fclose(this->file_ptr_);
        if (err_ != 0) {
            Fatal<std::ios_base::failure>("Cannot close file \"%s\".\n", this->filename_.c_str());
        }
    }
}

}  // namespace merlin
