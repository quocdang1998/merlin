// Copyright 2022 quocdang1998
#include "merlin/array/stock.hpp"

#include <chrono>  // std::chrono::milliseconds
#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <filesystem>  // std::filesystem::exists
#include <functional>  // std::bind, std::placeholders, std::ref
#include <thread>  // std::this_thread::sleep_for

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/copy.hpp"  // merlin::array::array_copy, merlin::array::contiguous_strides
#include "merlin/logger.hpp"  // WARNING, FAILURE
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin::array {

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

// Stock mutex
std::mutex Stock::mutex_;

// Convert char to openmode
std::ios_base::openmode Stock::char_to_openmode(char mode) {
    switch (mode) {
      case 'r':
        return std::ios_base::in;
      case 'w':
        return std::ios_base::out;
      case 'a':
        return std::ios_base::in | std::ios_base::out;
      default:
        FAILURE(std::invalid_argument, "Unknown openmode \"%c\".\n", mode);
    }
    return std::ios_base::binary;
}

// Get metadata from std::fstream
void Stock::get_fstream_metadata(void) {
    if (!(this->file_stream_.is_open())) {
        FAILURE(std::runtime_error, "File must be opened to get its metadata.\n");
    }
    this->stream_pos_ = this->file_stream_.tellg();
    this->format_flag_ = this->file_stream_.flags();
}

// Reopen a closed fstream
void Stock::reopen_fstream(void) {
    if (this->file_stream_.is_open()) {
        return;
    }
    this->file_stream_.open(this->filename_, Stock::char_to_openmode(this->mode_));
    this->file_stream_.seekg(this->stream_pos_);
    this->file_stream_.flags(this->format_flag_);
}

// Constructor from filename
Stock::Stock(const std::string & filename, char mode) : filename_(filename), mode_(mode) {
    this->file_stream_.open(filename, Stock::char_to_openmode(mode));
    if (!(this->file_stream_.is_open())) {
        if ((mode == 'r') || (mode == 'a') && !(std::filesystem::exists(filename))) {
            FAILURE(std::filesystem::filesystem_error, "Failed to open file %s (file doesn't exist).\n", filename.c_str());
        } else {
            do {
                WARNING("Cannot open file to write for now, refreshing in 1 minute.\n");
                std::this_thread::sleep_for(std::chrono::minutes(1));
            } while (!(this->file_stream_.is_open()));
        }
    }
    this->get_fstream_metadata();
}

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

// Destructor
Stock::~Stock(void) {
    if (this->file_stream_.is_open()) {
        this->file_stream_.close();
    }
}

}  // namespace merlin::array
