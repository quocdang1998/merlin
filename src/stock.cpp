// Copyright 2022 quocdang1998
#include "merlin/stock.hpp"

#include <mutex>  // std::mutex
#include <functional>  // std::bind, std::placeholders, std::ref

#include "merlin/vector.hpp"  // merlin::intvec
#include "merlin/array.hpp"  // merlin::Array
#include "merlin/utils.hpp"  // merlin::array_copy

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// Data read/write
// -------------------------------------------------------------------------------------------------------------------------

static inline void read_from_file(float * dest, std::fstream & file, float * src, unsigned long int count) {
    file.seekg(reinterpret_cast<uintptr_t>(src));
    file.read(reinterpret_cast<char *>(dest), count);
}

static inline void write_to_file(std::fstream & file, float * dest, float * src, unsigned long int count) {
    file.seekg(reinterpret_cast<uintptr_t>(dest));
    file.write(reinterpret_cast<char *>(src), count);
}

// -------------------------------------------------------------------------------------------------------------------------
// Stock
// -------------------------------------------------------------------------------------------------------------------------

// Constructor from filename
Stock::Stock(const std::string & filename) {
    this->file_stream_.open(filename, std::ios_base::in | std::ios_base::out);
    this->file_stream_.read(reinterpret_cast<char *>(&(this->ndim_)), sizeof(this->ndim_));
    this->shape_ = intvec(this->ndim_, 0);
    this->file_stream_.read(reinterpret_cast<char *>(this->shape_.data()), this->ndim_*sizeof(unsigned long int));
    this->data_ = reinterpret_cast<float *>(static_cast<uintptr_t>(this->file_stream_.tellg()));
}

// Copy data from Stock to Array
Array Stock::to_array(void) {
    // allocate result
    Array result(this->shape_);
    // construct read function
    auto read_func = std::bind(read_from_file, std::placeholders::_1, std::ref(this->file_stream_), std::placeholders::_2, std::placeholders::_3);
    // copy data in between a mutex lock to prevent data races
    std::mutex m;
    m.lock();
    array_copy(&result, this, read_func);
    m.unlock();
    return result;
}

void Stock::dumped(const Array & src) {
    // construct write function
    auto write_func = std::bind(write_to_file, std::ref(this->file_stream_), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    // copy data in between a mutex lock to prevent data races
    std::mutex m;
    m.lock();
    array_copy(this, &src, write_func);
    m.unlock();
}

// Destructor
Stock::~Stock(void) {
    this->file_stream_.close();
}

}  // namespace merlin
