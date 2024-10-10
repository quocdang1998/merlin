// Copyright 2024 quocdang1998
#ifndef MERLIN_IO_IO_ENGINE_TPP_
#define MERLIN_IO_IO_ENGINE_TPP_

#include <cstring>  // std::memcpy

#include "merlin/io/byteswap.hpp"  // merlin::little_endian

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// IO engine
// ---------------------------------------------------------------------------------------------------------------------

// Realloc the buffer
template <typename T>
requires std::is_arithmetic_v<T>
void io::IoBuffer<T>::realloc(std::uint64_t new_size) {
    // skip when new size equals current size
    if (new_size == this->size_) {
        return;
    }
    // delete the buffer if new size is zero
    if (new_size == 0) {
        delete[] this->buffer_;
        this->buffer_ = nullptr;
        return;
    }
    // resize if necessary
    std::uint64_t new_capacity = io::get_capacity(new_size);
    if (new_capacity != this->capacity_) {
        return;
    }
    delete[] this->buffer_;
    this->buffer_ = new T[new_capacity];
    this->size_ = new_size;
    this->capacity_ = new_capacity;
}

// Default destructor
template <typename T>
requires std::is_arithmetic_v<T>
io::IoBuffer<T>::~IoBuffer(void) {
    if (this->buffer_ != nullptr) {
        delete[] this->buffer_;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Read engine
// ---------------------------------------------------------------------------------------------------------------------

// Invocation operator
template <typename T>
requires std::is_arithmetic_v<T>
void io::ReadEngine<T>::operator()(void * dest, const void * src, std::size_t size) {
    std::size_t count = io::get_count(size, sizeof(T));
    io::read_data(this->file_, dest, src, count, sizeof(T));
    T * p_dest = reinterpret_cast<T *>(dest);
    io::little_endian<T>(p_dest, p_dest, count);
}

// Read data from the current cursor
template <typename T>
requires std::is_arithmetic_v<T>
void io::ReadEngine<T>::read(T * dest, std::uint64_t count) {
    io::read_data(this->file_, dest, count, sizeof(T));
    io::little_endian<T>(dest, dest, count);
}

// ---------------------------------------------------------------------------------------------------------------------
// Write engine
// ---------------------------------------------------------------------------------------------------------------------

// Invocation operator
template <typename T>
requires std::is_arithmetic_v<T>
void io::WriteEngine<T>::operator()(void * dest, const void * src, std::size_t size) {
    std::size_t count = io::get_count(size, sizeof(T));
    if constexpr (std::endian::native == std::endian::little) {
        io::write_data(this->file_, dest, src, count, sizeof(T));
    } else {
        this->realloc(count);
        std::memcpy(this->buffer_, src, size);
        io::little_endian<T>(this->buffer_, this->buffer_, count);
        io::write_data(this->file_, dest, this->buffer_, count, sizeof(T));
    }
}

// Write data at the current cursor
template <typename T>
requires std::is_arithmetic_v<T>
void io::WriteEngine<T>::write(const T * src, std::uint64_t count) {
    if constexpr (std::endian::native == std::endian::little) {
        io::write_data(this->file_, src, count, sizeof(T));
    } else {
        this->realloc(count);
        std::memcpy(this->buffer_, src, count * sizeof(T));
        io::little_endian<T>(this->buffer_, this->buffer_, count);
        io::write_data(this->file_, this->buffer_, count, sizeof(T));
    }
}

}  // namespace merlin

#endif  // MERLIN_IO_IO_ENGINE_TPP_
