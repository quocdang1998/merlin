// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <cinttypes>  // PRIu64
#include <functional>  // std::bind, std::placeholders
#include <sstream>  // std::ostringstream

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy, merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // WARNING, FAILURE
#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__
#include "merlin/utils.hpp"  // merlin::get_current_process_id, merlin::get_time
#include "merlin/vector.hpp"  // merlin::intvec

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
array::Stock::Stock(const std::string & filename, char mode, std::uint64_t offset, bool thread_safe) :
filename_(filename), offset_(offset), thread_safe_(thread_safe) {
    switch (mode) {
        case 'r':
            this->file_ptr_ = std::fopen(filename.c_str(), "rb");
            this->mode_ = 'r';
            break;
        case 'w':
            // crash old content if file exist
            {
                std::FILE * temporary = std::fopen(filename.c_str(), "wb");
                std::fclose(temporary);
            }
            this->file_ptr_ = std::fopen(filename.c_str(), "rb+");
            this->mode_ = 'a';
            break;
        case 'a':
            this->file_ptr_ = std::fopen(filename.c_str(), "rb+");
            this->mode_ = 'a';
            break;
        default:
            FAILURE(std::ios_base::failure, "Unknown open mode %c.\n", mode);
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

// Construct empty Array from shape vector
array::Stock::Stock(const std::string & filename, const intvec & shape, bool thread_safe) : array::NdData(shape),
filename_(filename), mode_('a'), offset_(0), thread_safe_(thread_safe) {
    // initialize data
    std::FILE * temporary = std::fopen(filename.c_str(), "wb");
    std::fclose(temporary);
    this->file_ptr_ = std::fopen(filename.c_str(), "rb+");
    this->flock_ = FileLock(this->file_ptr_);
    // allocate file memory
    this->write_metadata();
    if (thread_safe) {
        this->flock_.lock();
    }
    std::uintptr_t cursor = std::ftell(this->file_ptr_);
    this->data_ = reinterpret_cast<float *>(cursor);
    std::fseek(this->file_ptr_, cursor + (this->size() * sizeof(float)) - 1, SEEK_SET);
    std::fputc('\0', this->file_ptr_);
    if (thread_safe) {
        this->flock_.unlock();
    }
}

// Get value of element at a n-dim index
float array::Stock::get(const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    float result;
    if (this->thread_safe_) {
        this->flock_.lock_shared();
    }
    read_from_file(&result, this->file_ptr_, reinterpret_cast<float *>(data_ptr), sizeof(float));
    if (this->thread_safe_) {
        this->flock_.unlock();
    }
    return result;
}

// Get value of element at a C-contiguous index
float array::Stock::get(std::uint64_t index) const {
    return this->get(contiguous_to_ndim_idx(index, this->shape()));
}

// Set value of element at a n-dim index
void array::Stock::set(const intvec index, float value) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    if (this->thread_safe_) {
        this->flock_.lock();
    }
    write_to_file(this->file_ptr_, reinterpret_cast<float *>(data_ptr), &value, sizeof(float));
    if (this->thread_safe_) {
        this->flock_.unlock();
    }
}

// Set value of element at a C-contiguous index
void array::Stock::set(std::uint64_t index, float value) {
    this->set(contiguous_to_ndim_idx(index, this->shape()), value);
}

// Read metadata from file (this function is single threaded to avoid data race)
void array::Stock::read_metadata(void) {
    if (this->thread_safe_) {
        this->flock_.lock_shared();
    }
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    std::fread(&(this->ndim_), sizeof(std::uint64_t), 1, this->file_ptr_);
    this->shape_ = intvec(this->ndim_, 0);
    std::fread(this->shape_.data(), sizeof(std::uint64_t), this->ndim_, this->file_ptr_);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    this->data_ = reinterpret_cast<float *>(std::uintptr_t(std::ftell(this->file_ptr_)));
    if (this->thread_safe_) {
        this->flock_.unlock();
    }
}

// Copy data from file to an array
void array::Stock::copy_to_array(array::Array & arr) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, this->file_ptr_,
                               std::placeholders::_2, std::placeholders::_3);
    if (this->thread_safe_) {
        this->flock_.lock_shared();
    }
    array::array_copy(&arr, this, read_func);
    if (this->thread_safe_) {
        this->flock_.unlock();
    }
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
    if (this->thread_safe_) {
        this->flock_.lock();
    }
    std::fseek(this->file_ptr_, this->offset_, SEEK_SET);
    std::fwrite(&(this->ndim_), sizeof(std::uint64_t), 1, this->file_ptr_);
    std::fwrite(this->shape_.data(), sizeof(std::uint64_t), this->ndim_, this->file_ptr_);
    if (this->thread_safe_) {
        this->flock_.unlock();
    }
}

// Write data from an array to a file
void array::Stock::write_data_to_file(array::Array & src) {
    auto write_func = std::bind(write_to_file, this->file_ptr_,
                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    if (this->thread_safe_) {
        this->flock_.lock();
    }
    array_copy(this, &src, write_func);
    if (this->thread_safe_) {
        this->flock_.unlock();
    }
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
