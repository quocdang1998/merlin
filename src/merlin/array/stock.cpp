// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <cinttypes>   // PRIu64
#include <filesystem>  // std::filesystem::filesystem_error, std::filesystem::file_size, std::filesystem::resize_file
#include <functional>  // std::bind, std::placeholders
#include <ios>         // std::ios_base::failure

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/logger.hpp"           // WARNING, FAILURE
#include "merlin/platform.hpp"         // __MERLIN_LINUX__, __MERLIN_WINDOWS__
#include "merlin/utils.hpp"            // merlin::get_current_process_id, merlin::get_time

// Acquire lockfile if thread safe
#define EXCLUSIVE_LOCK_THREADSAFE()                                                                                    \
    if (this->thread_safe_) this->flock_.lock()

// Acquire shared lockfile if thread safe
#define SHARED_LOCK_THREADSAFE()                                                                                       \
    if (this->thread_safe_) this->flock_.lock_shared()

// Unlock file if thread safe
#define UNLOCK_THREADSAFE()                                                                                            \
    if (this->thread_safe_) this->flock_.unlock()

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Data read/write
// ---------------------------------------------------------------------------------------------------------------------

// Read an array from file
static inline void read_from_file(double * dest, std::FILE * file, double * src, std::uint64_t bytes) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (std::fread(dest, sizeof(double), count, file) != count) {
        FAILURE(std::ios_base::failure, "Read file error.\n");
    }
}

// Write an array from file
static inline void write_to_file(std::FILE * file, double * dest, double * src, std::uint64_t bytes) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(dest), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (std::fwrite(src, sizeof(double), bytes / sizeof(double), file) != count) {
        FAILURE(std::ios_base::failure, "Write file error.\n");
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
    // zero fill shape and stride
    this->shape_.fill(0);
    this->strides_.fill(0);
    // read ndim and shape data from file at position offset
    SHARED_LOCK_THREADSAFE();
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    if (std::fread(&(this->ndim_), sizeof(std::uint64_t), 1, this->file_ptr_) != 1) {
        FAILURE(std::ios_base::failure, "Read file error.\n");
    }
    if (std::fread(this->shape_.data(), sizeof(std::uint64_t), this->ndim_, this->file_ptr_) != this->ndim_) {
        FAILURE(std::ios_base::failure, "Read file error.\n");
    }
    std::uint64_t cursor = std::ftell(this->file_ptr_);
    UNLOCK_THREADSAFE();
    // calculate stride and assign data pointer
    this->calc_array_size();
    this->strides_ = array::contiguous_strides(this->shape_, this->ndim_, sizeof(double));
    this->data_ = reinterpret_cast<double *>(cursor);
    // check file size
    std::uint64_t file_size = std::filesystem::file_size(this->filename_);
    std::uint64_t expected_size = this->offset_ + (1 + this->ndim_) * sizeof(std::uint64_t);
    expected_size += this->size() * sizeof(double);
    if (file_size < expected_size) {
        FAILURE(std::filesystem::filesystem_error, "Expected filesize of at least %" PRIu64 ", got %" PRIu64 ".\n",
                expected_size, file_size);
    }
    return cursor;
}

// Write metadata to file at offset position
std::uint64_t array::Stock::write_metadata(void) {
    // write ndim and shape data to file at position offset
    std::uint64_t ndim = this->ndim();
    EXCLUSIVE_LOCK_THREADSAFE();
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    if (std::fwrite(&ndim, sizeof(std::uint64_t), 1, this->file_ptr_) != 1) {
        FAILURE(std::ios_base::failure, "Write file error.\n");
    }
    if (std::fwrite(this->shape_.data(), sizeof(std::uint64_t), ndim, this->file_ptr_) != ndim) {
        FAILURE(std::ios_base::failure, "Write file error.\n");
    }
    std::uint64_t cursor = std::ftell(this->file_ptr_);
    UNLOCK_THREADSAFE();
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
        FAILURE(std::filesystem::filesystem_error, "Cannot find file \"%s\", please make sure that the file exists.\n",
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
    double result;
    SHARED_LOCK_THREADSAFE();
    read_from_file(&result, this->file_ptr_, reinterpret_cast<double *>(data_ptr), sizeof(double));
    UNLOCK_THREADSAFE();
    return result;
}

// Get value of element at a C-contiguous index
double array::Stock::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    double result;
    SHARED_LOCK_THREADSAFE();
    read_from_file(&result, this->file_ptr_, reinterpret_cast<double *>(data_ptr), sizeof(double));
    UNLOCK_THREADSAFE();
    return result;
}

// Set value of element at a n-dim index
void array::Stock::set(const Index & index, double value) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    EXCLUSIVE_LOCK_THREADSAFE();
    write_to_file(this->file_ptr_, reinterpret_cast<double *>(data_ptr), &value, sizeof(double));
    UNLOCK_THREADSAFE();
}

// Set value of element at a C-contiguous index
void array::Stock::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_, this->strides_, this->ndim_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    EXCLUSIVE_LOCK_THREADSAFE();
    write_to_file(this->file_ptr_, reinterpret_cast<double *>(data_ptr), &value, sizeof(double));
    UNLOCK_THREADSAFE();
}

// Set value of all elements
void array::Stock::fill(double value) {
    auto write_func = std::bind(write_to_file, this->file_ptr_, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3);
    array::fill(this, value, write_func);
}

// Write data from an array to a file
void array::Stock::record_data_to_file(const array::Array & src) {
    auto write_func = std::bind(write_to_file, this->file_ptr_, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3);
    EXCLUSIVE_LOCK_THREADSAFE();
    copy(this, &src, write_func);
    UNLOCK_THREADSAFE();
}

// String representation
std::string array::Stock::str(bool first_call) const { return array::print(this, "Stock", first_call); }

// Destructor
array::Stock::~Stock(void) {
    if (this->release && (this->file_ptr_ != nullptr)) {
        int err_ = std::fclose(this->file_ptr_);
        if (err_ != 0) {
            FAILURE(std::ios_base::failure, "Cannot close file \"%s\".\n", this->filename_.c_str());
        }
    }
}

}  // namespace merlin
