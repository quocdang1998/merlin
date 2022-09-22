// Copyright 2022 quocdang1998
#include "merlin/stock.hpp"

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <functional>  // std::bind, std::placeholders, std::ref

#include "merlin/vector.hpp"  // merlin::intvec
#include "merlin/array.hpp"  // merlin::Array
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::array_copy, merlin::contiguous_strides

namespace merlin {

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

// Constructor from filename
Stock::Stock(const std::string & filename) {
    this->file_stream_.open(filename, std::ios_base::in | std::ios_base::out);
}

// Read metadata from file (this function is single threaded to avoid data race)
void Stock::read_metadata(void) {
    Stock::mutex_.lock();
    this->file_stream_.seekg(0, std::ios_base::beg);
    this->file_stream_.read(reinterpret_cast<char *>(&(this->ndim_)), sizeof(std::uint64_t));
    this->shape_ = intvec(this->ndim_, 0);
    this->file_stream_.read(reinterpret_cast<char *>(this->shape_.data()), this->ndim_*sizeof(std::uint64_t));
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    this->data_ = reinterpret_cast<float *>(std::uintptr_t(this->file_stream_.tellg()));
    Stock::mutex_.unlock();
}

// Copy data from file to an array
void Stock::copy_to_array(Array & arr) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, std::ref(this->file_stream_),
                               std::placeholders::_2, std::placeholders::_3);
    Stock::mutex_.lock();
    array_copy(&arr, this, read_func);
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
    this->file_stream_.seekg(0, std::ios_base::beg);
    this->file_stream_.write(reinterpret_cast<char *>(&(this->ndim_)), sizeof(std::uint64_t));
    this->file_stream_.write(reinterpret_cast<char *>(this->shape_.data()), this->ndim_*sizeof(std::uint64_t));
    this->data_ = reinterpret_cast<float *>(std::uintptr_t(this->file_stream_.tellg()));
    Stock::mutex_.unlock();
}

// Write data from an array to a file
void Stock::write_data_to_file(Array & src) {
    auto write_func = std::bind(write_to_file, std::ref(this->file_stream_),
                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    Stock::mutex_.lock();
    array_copy(this, &src, write_func);
    Stock::mutex_.unlock();
}

// Destructor
Stock::~Stock(void) {
    this->file_stream_.close();
}

}  // namespace merlin
