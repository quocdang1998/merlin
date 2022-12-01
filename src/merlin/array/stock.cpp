// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <cinttypes>  // PRIu64
#include <functional>  // std::bind, std::placeholders

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy, merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // WARNING, FAILURE
#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__
#include "merlin/vector.hpp"  // merlin::intvec

#if defined(__MERLIN_WINDOWS__)
#include <share.h>  // _SH_DENYNO, _SH_DENYRW, _SH_DENYWR
#endif  // __MERLIN_WINDOWS__

// --------------------------------------------------------------------------------------------------------------------
// Open file pointer (Windows)
// --------------------------------------------------------------------------------------------------------------------

#if defined(__MERLIN_WINDOWS__)

// Open file for read (write access denied)
static inline std::FILE * read_file(const char * fname) {
    return ::_fsopen(fname, "rb", _SH_DENYWR);
}

// Open file for write (read and write access denied)
static inline std::FILE * write_file(const char * fname, bool thread_safe = true) {
    int sh_flag = thread_safe ? _SH_DENYRW : _SH_DENYNO;
    std::FILE * temporary = std::fopen(fname, "wb");  // crash old content
    std::fclose(temporary);
    return ::_fsopen(fname, "rb+", sh_flag);
}

// Open file for read and write (read and write access denied)
static inline std::FILE * update_file(const char * fname, bool thread_safe = true) {
    int sh_flag = thread_safe ? _SH_DENYRW : _SH_DENYNO;
    return ::_fsopen(fname, "rb+", sh_flag);
}

#endif  // __MERLIN_WINDOWS__

// --------------------------------------------------------------------------------------------------------------------
// Open file pointer (Linux)
// --------------------------------------------------------------------------------------------------------------------

#if defined(__MERLIN_LINUX__)

// Open file for read (write access denied)
static inline std::FILE * read_file(const char * fname) {
    return std::fopen(fname, "rb");
}

// Open file for write (read and write access denied)
static inline std::FILE * write_file(const char * fname, bool thread_safe = true) {
    std::FILE * temporary = std::fopen(fname, "wb");  // crash old content
    std::fclose(temporary);
    return std::fopen(fname, "rb+");
}

// Open file for read and write (read and write access denied)
static inline std::FILE * update_file(const char * fname, bool thread_safe = true) {
    return std::fopen(fname, "rb+");
}

#endif  // __MERLIN_LINUX__

// --------------------------------------------------------------------------------------------------------------------
// Data read/write
// --------------------------------------------------------------------------------------------------------------------

// Read an array from file
static inline void read_from_file(float * dest, std::FILE * file, float * src, std::uint64_t bytes) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET);
    std::fread(dest, sizeof(float), bytes / sizeof(float), file);
}

static inline void write_to_file(std::FILE * file, float * dest, float * src, std::uint64_t bytes) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(dest), SEEK_SET);
    std::fwrite(src, sizeof(float), bytes / sizeof(float), file);
}

// --------------------------------------------------------------------------------------------------------------------
// Stock
// --------------------------------------------------------------------------------------------------------------------

namespace merlin {

// Constructor from filename
array::Stock::Stock(const std::string & filename, char mode, std::uint64_t offset) :
filename_(filename), mode_(mode), offset_(offset) {
    switch (mode) {
        case 'r':
            this->file_ptr_ = read_file(filename.c_str());
            break;
        case 'w':
            this->file_ptr_ = write_file(filename.c_str(), true);
            break;
        case 'a':
            this->file_ptr_ = update_file(filename.c_str(), true);
            break;
        case 'p':
            this->file_ptr_ = write_file(filename.c_str(), false);
            break;
        case 's':
            this->file_ptr_ = update_file(filename.c_str(), false);
            break;
    }
    if (this->file_ptr_ == nullptr) {
        FAILURE(std::ios_base::failure, "Cannot open file \"%s\".\n", filename.c_str());
    }
    this->flock_ = FileLock(this->file_ptr_);
    int err_ = std::fseek(this->file_ptr_, offset, SEEK_SET);
    if (err_ != 0) {
        FAILURE(std::ios_base::failure, "Cannot move cursor to position %" PRIu64 " to file \"%s\".\n",
                offset, filename.c_str());
    }
}

// Read metadata from file (this function is single threaded to avoid data race)
void array::Stock::read_metadata(void) {
    this->flock_.lock_shared();
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    std::fread(&(this->ndim_), sizeof(std::uint64_t), 1, this->file_ptr_);
    this->shape_ = intvec(this->ndim_, 0);
    std::fread(this->shape_.data(), sizeof(std::uint64_t), this->ndim_, this->file_ptr_);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    this->data_ = reinterpret_cast<float *>(std::uintptr_t(std::ftell(this->file_ptr_)));
    this->flock_.unlock();
}

// Copy data from file to an array
void array::Stock::copy_to_array(array::Array & arr) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, this->file_ptr_,
                               std::placeholders::_2, std::placeholders::_3);
    this->flock_.lock_shared();
    array::array_copy(&arr, this, read_func);
    this->flock_.unlock();
}

// Copy data from Stock to Array
array::Array array::Stock::to_array(void) {
    // read metadata
    this->read_metadata();
    // allocate result
    Array result(this->shape_);
    // copy data
    this->copy_to_array(result);
    return result;
}

// Get metadata from an array
void array::Stock::get_metadata(array::Array & src) {
    this->ndim_ = src.ndim();
    this->shape_ = src.shape();
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    this->data_ = reinterpret_cast<float *>(this->offset_ + sizeof(std::uint64_t)*(1+this->ndim_));
}

// Write metadata to file
void array::Stock::write_metadata(void) {
    this->flock_.lock();
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    std::fwrite(&(this->ndim_), sizeof(std::uint64_t), 1, this->file_ptr_);
    std::fwrite(this->shape_.data(), sizeof(std::uint64_t), this->ndim_, this->file_ptr_);
    this->flock_.unlock();
}

// Write data from an array to a file
void array::Stock::write_data_to_file(array::Array & src) {
    auto write_func = std::bind(write_to_file, this->file_ptr_,
                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    this->flock_.lock();
    array_copy(this, &src, write_func);
    this->flock_.unlock();
}

// Destructor
array::Stock::~Stock(void) {
    if (this->file_ptr_ != nullptr) {
        int err_ = std::fclose(this->file_ptr_);
        if (err_ != 0) {
            FAILURE(std::ios_base::failure, "Cannot close file \"%s\".\n", this->filename_.c_str());
        }
    }
}

}  // namespace merlin
