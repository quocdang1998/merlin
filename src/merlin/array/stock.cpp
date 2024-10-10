// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <cinttypes>     // PRIu64
#include <filesystem>    // std::filesystem::filesystem_error, std::filesystem::file_size, std::filesystem::resize_file
#include <ios>           // std::ios_base::failure
#include <mutex>         // std::unique_lock
#include <shared_mutex>  // std::shared_lock

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::contiguous_strides, merlin::array::get_leap,
                                       // merlin::array::copy, merlin::array::fill, merlin::array::print
#include "merlin/io/io_engine.hpp"     // merlin::io::ReadEngine, merlin::io::WriteEngine
#include "merlin/logger.hpp"           // merlin::Fatal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Data read/write
// ---------------------------------------------------------------------------------------------------------------------

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
    // initialize lock guard and reader
    std::shared_lock<io::FileLock> lock = ((this->thread_safe_) ? std::shared_lock<io::FileLock>(this->flock_)
                                                                : std::shared_lock<io::FileLock>());
    io::ReadEngine<std::uint64_t> reader(this->file_ptr_);
    // move cursor to offset
    if (std::fseek(this->file_ptr_, this->offset_, SEEK_SET)) {
        Fatal<std::ios_base::failure>("Seek file error.\n");
    }
    // read data shape
    std::uint64_t ndim;
    reader.read(&ndim, 1);
    this->shape_.resize(ndim);
    reader.read(this->shape_.data(), this->shape_.size());
    std::uint64_t cursor = std::ftell(this->file_ptr_);
    // calculate stride and assign data pointer
    this->calc_array_size();
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    this->data_ = reinterpret_cast<double *>(cursor);
    // check file size
    std::uint64_t file_size = std::filesystem::file_size(this->filename_);
    std::uint64_t expected_size = this->offset_ + (1 + ndim) * sizeof(std::uint64_t) + this->size() * sizeof(double);
    if (file_size < expected_size) {
        Fatal<std::filesystem::filesystem_error>("Expected filesize of at least %" PRIu64 ", got %" PRIu64 ".\n",
                                                 expected_size, file_size);
    }
    return cursor;
}

// Write metadata to file at offset position
std::uint64_t array::Stock::write_metadata(void) {
    // initialize lock guard and writer
    std::unique_lock<io::FileLock> lock = ((this->thread_safe_) ? std::unique_lock<io::FileLock>(this->flock_)
                                                                : std::unique_lock<io::FileLock>());
    io::WriteEngine<std::uint64_t> writer(this->file_ptr_);
    // move cursor to offset
    if (std::fseek(this->file_ptr_, this->offset_, SEEK_SET)) {
        Fatal<std::ios_base::failure>("Seek file error.\n");
    }
    // write shape data
    std::uint64_t ndim = this->ndim();
    writer.write(&ndim, 1);
    writer.write(this->shape_.data(), this->shape_.size());
    std::uint64_t cursor = std::ftell(this->file_ptr_);
    // change data pointer to current cursor
    this->data_ = reinterpret_cast<double *>(cursor);
    return cursor;
}

// Construct empty Array from shape vector
array::Stock::Stock(const std::string & filename, const Index & shape, std::uint64_t offset, bool thread_safe) :
array::NdData(shape), filename_(filename), offset_(offset), thread_safe_(thread_safe) {
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
    this->flock_ = io::FileLock(this->file_ptr_);
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
    this->flock_ = io::FileLock(this->file_ptr_);
    this->read_metadata();
    this->release = true;
}

// Get value of element at a n-dim index
double array::Stock::get(const Index & index) const {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::shared_lock<io::FileLock> lock = ((this->thread_safe_) ? std::shared_lock<io::FileLock>(this->flock_)
                                                                : std::shared_lock<io::FileLock>());
    io::ReadEngine<double> reader(this->file_ptr_);
    double result;
    reader(&result, reinterpret_cast<double *>(data_ptr), sizeof(double));
    return result;
}

// Get value of element at a C-contiguous index
double array::Stock::get(std::uint64_t index) const {
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::shared_lock<io::FileLock> lock = ((this->thread_safe_) ? std::shared_lock<io::FileLock>(this->flock_)
                                                                : std::shared_lock<io::FileLock>());
    io::ReadEngine<double> reader(this->file_ptr_);
    double result;
    reader(&result, reinterpret_cast<double *>(data_ptr), sizeof(double));
    return result;
}

// Set value of element at a n-dim index
void array::Stock::set(const Index & index, double value) {
    std::uint64_t leap = inner_prod(index.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::unique_lock<io::FileLock> lock = ((this->thread_safe_) ? std::unique_lock<io::FileLock>(this->flock_)
                                                                : std::unique_lock<io::FileLock>());
    io::WriteEngine<double> writer(this->file_ptr_);
    writer(reinterpret_cast<double *>(data_ptr), &value, sizeof(double));
}

// Set value of element at a C-contiguous index
void array::Stock::set(std::uint64_t index, double value) {
    std::uint64_t leap = array::get_leap(index, this->shape_.data(), this->strides_.data(), this->ndim());
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    std::unique_lock<io::FileLock> lock = ((this->thread_safe_) ? std::unique_lock<io::FileLock>(this->flock_)
                                                                : std::unique_lock<io::FileLock>());
    io::WriteEngine<double> writer(this->file_ptr_);
    writer(reinterpret_cast<double *>(data_ptr), &value, sizeof(double));
}

// Set value of all elements
void array::Stock::fill(double value) {
    std::unique_lock<io::FileLock> lock = ((this->thread_safe_) ? std::unique_lock<io::FileLock>(this->flock_)
                                                                : std::unique_lock<io::FileLock>());
    io::WriteEngine<double> writer(this->file_ptr_);
    array::fill(this, value, writer);
}

// Calculate mean and variance of all non-zero and finite elements
std::array<double, 2> array::Stock::get_mean_variance(void) const {
    std::shared_lock<io::FileLock> lock = ((this->thread_safe_) ? std::shared_lock<io::FileLock>(this->flock_)
                                                                : std::shared_lock<io::FileLock>());
    io::ReadEngine<double> reader(this->file_ptr_);
    return array::stat(this, reader);
}

// Write data from an array to a file
void array::Stock::record_data_to_file(const array::Array & src) {
    std::unique_lock<io::FileLock> lock = ((this->thread_safe_) ? std::unique_lock<io::FileLock>(this->flock_)
                                                                : std::unique_lock<io::FileLock>());
    io::WriteEngine<double> writer(this->file_ptr_);
    copy(this, &src, writer);
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
