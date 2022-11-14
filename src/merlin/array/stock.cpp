// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <chrono>  // std::chrono::milliseconds
#include <cinttypes>  // PRIu64
#include <ios>  // std::ios_base::failure
#include <filesystem>  // std::filesystem::exists
#include <functional>  // std::bind, std::placeholders, std::ref
#include <thread>  // std::this_thread::sleep_for

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif  // WIN32 || _WIN32 || WIN64 || _WIN64

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy, merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // WARNING, FAILURE
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin::array {

// -------------------------------------------------------------------------------------------------------------------------
// FileLock
// -------------------------------------------------------------------------------------------------------------------------

// Constructor from filename
FileLock::FileLock(const char * fname) {
    this->file_handle = CreateFileA(fname, GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, NULL);
}

// Lock file handle
void FileLock::lock(void) {
    bool blocked = LockFile(this->file_handle, 0, 0, 1024, 0);
}

// Destructor
FileLock::~FileLock(void) {
    bool err_ = CloseHandle(this->file_handle);
}

// -------------------------------------------------------------------------------------------------------------------------
// Data read/write
// -------------------------------------------------------------------------------------------------------------------------

static inline void read_from_file(float * dest, std::fstream & file, float * src, std::uint64_t count) {
    file.seekg(reinterpret_cast<std::uintptr_t>(src));
    file.read(reinterpret_cast<char *>(dest), count);
}

static inline void write_to_file(std::fstream & file, float * dest, float * src, std::uint64_t count) {
    file.seekg(reinterpret_cast<std::uintptr_t>(dest));
    file.write(reinterpret_cast<char *>(src), count);
}

// -------------------------------------------------------------------------------------------------------------------------
// Stock
// -------------------------------------------------------------------------------------------------------------------------

// Constructor from filename
Stock::Stock(const std::string & filename, char mode, std::uint64_t offset) : filename_(filename), mode_(mode),
                                                                              offset_(offset) {
    std::string c_mode;
    switch (mode) {
        case 'r':
            c_mode = std::string("rb");
            break;
        case 'w', 'p':
            c_mode = std::string("wb");
            break;
        case 'a', 's':
            c_mode = std::string("rb+");
            break;
    }
    this->file_ptr_ = std::fopen(filename.c_str(), c_mode.c_str());
    if (this->file_ptr_ == NULL) {
        FAILURE(std::ios_base::failure, "Cannot open file \"%s\".\n", filename.c_str());
    }
    int err_ = std::fseek(this->file_ptr_, offset, SEEK_SET);
    if (err_ != 0) {
        FAILURE(std::ios_base::failure, "Cannot move cursor to position %" PRIu64 " to file \"%s\".\n",
                offset, filename.c_str());
    }
}

// Temporary close the file
void Stock::temporary_close(void) {
    if (this->file_ptr_ != NULL) {
        int err_ = std::fclose(this->file_ptr_);
        if (err_ != 0) {
            FAILURE(std::ios_base::failure, "Cannot close file \"%s\".\n", this->filename_.c_str());
        }
        this->file_ptr_ = NULL;
    }
}

/*
// Read metadata from file (this function is single threaded to avoid data race)
void Stock::read_metadata(void) {
    Stock::mutex_.lock();
    this->reopen_fstream();
    this->file_stream_.seekg(0, std::ios_base::beg);
    this->file_stream_.read(reinterpret_cast<char *>(&(this->ndim_)), sizeof(std::uint64_t));
    this->shape_ = intvec(this->ndim_, 0);
    this->file_stream_.read(reinterpret_cast<char *>(this->shape_.data()), this->ndim_*sizeof(std::uint64_t));
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    this->data_ = reinterpret_cast<float *>(std::uintptr_t(this->file_stream_.tellg()));
    this->file_stream_.close();
    Stock::mutex_.unlock();
}

// Copy data from file to an array
void Stock::copy_to_array(Array & arr) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, std::ref(this->file_stream_),
                               std::placeholders::_2, std::placeholders::_3);
    Stock::mutex_.lock();
    this->reopen_fstream();
    array_copy(&arr, this, read_func);
    this->file_stream_.close();
    Stock::mutex_.unlock();
}

// Copy data from Stock to Array
Array Stock::to_array(void) {
    // read metadata
    this->read_metadata();
    // allocate result
    Array result(this->shape_);
    // copy data
    this->copy_to_array(result);
    return result;
}

// Get metadata from an array
void Stock::get_metadata(Array & src) {
    this->ndim_ = src.ndim();
    this->shape_ = src.shape();
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
}

// Write metadata to file
void Stock::write_metadata(void) {
    Stock::mutex_.lock();
    // this->reopen_fstream();
    this->file_stream_.seekg(0, std::ios_base::beg);
    this->file_stream_.write(reinterpret_cast<char *>(&(this->ndim_)), sizeof(std::uint64_t));
    this->file_stream_.write(reinterpret_cast<char *>(this->shape_.data()), this->ndim_*sizeof(std::uint64_t));
    this->data_ = reinterpret_cast<float *>(std::uintptr_t(this->file_stream_.tellg()));
    // this->file_stream_.close();
    Stock::mutex_.unlock();
}

// Write data from an array to a file
void Stock::write_data_to_file(Array & src) {
    auto write_func = std::bind(write_to_file, std::ref(this->file_stream_),
                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    Stock::mutex_.lock();
    // this->reopen_fstream();
    array_copy(this, &src, write_func);
    // this->file_stream_.close();
    Stock::mutex_.unlock();
}
*/
// Destructor
Stock::~Stock(void) {
    if ((this->file_ptr_ != NULL) && this->force_close) {
        int err_ = std::fclose(this->file_ptr_);
        if (err_ != 0) {
            FAILURE(std::ios_base::failure, "Cannot close file \"%s\".\n", this->filename_.c_str());
        }
    }
}

}  // namespace merlin::array
