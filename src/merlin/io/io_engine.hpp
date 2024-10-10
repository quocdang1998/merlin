// Copyright 2024 quocdang1998
#ifndef MERLIN_IO_IO_ENGINE_HPP_
#define MERLIN_IO_IO_ENGINE_HPP_

#include <cstddef>      // nullptr, std::size_t
#include <cstdint>      // std::uint64_t
#include <cstdio>       // std::FILE
#include <type_traits>  // std::add_pointer, std::is_arithmetic_v

#include "merlin/io/declaration.hpp"   // merlin::io::IoBuffer, merlin::io::ReadEngine, merlin::io::WriteEngine

namespace merlin {

// Utility
// -------

namespace io {

/** @brief Get the smallest power of 2 that is greater than a number.*/
constexpr std::uint64_t get_capacity(std::uint64_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

/** @brief Get the number of element.
 *  @details This function would throw an error if the size is not a multiple of element size.
 */
std::size_t get_count(std::uint64_t size, std::size_t element_size);

/** @brief Read data from file at the current cursor.*/
void read_data(std::FILE * file, void * dest, std::size_t count, std::size_t element_size);

/** @brief Read data from file at a specified location.
 *  @return The number of elements successfully read.
 */
void read_data(std::FILE * file, void * dest, const void * src, std::size_t count, std::size_t element_size);

/** @brief Write data to file at the current cursor.*/
void write_data(std::FILE * file, const void * src, std::size_t count, std::size_t element_size);

/** @brief Write data to file at a specified location.
 *  @return The number of elements successfully read.
 */
void write_data(std::FILE * file, void * dest, const void * src, std::size_t count, std::size_t element_size);

}  // namespace io

// IO engine
// ---------

/** @brief Base engine.
 *  @details Buffer for input/output.
 */
template <typename T>
requires std::is_arithmetic_v<T>
class io::IoBuffer {
  public:
    /** @brief Constructor from file pointer.*/
    IoBuffer(std::FILE * fp) : file_(fp) {}
    /** @brief Realloc the buffer.*/
    void realloc(std::uint64_t new_size);
    /** @brief Invocation operator.*/
    virtual void operator()(void * dest, const void * src, std::size_t size) = 0;
    /** @brief Get reference to the file pointer.*/
    constexpr std::FILE * get_fp(void) noexcept { return this->file_; }
    /** @brief Default destructor.*/
    ~IoBuffer(void);

  protected:
    /** @brief Buffer.*/
    T * buffer_ = nullptr;
    /** @brief Size of the buffer.*/
    std::uint64_t size_ = 0;
    /** @brief Size of the allocated memory for the buffer.*/
    std::uint64_t capacity_ = 0;
    /** @brief Reference to the file pointer.*/
    std::FILE * file_;
};

// Read engine
// -----------

/** @brief Reading engine.
 *  @details Read an array from a binary file written in little endian. The engine also performs checking for reading
 *  errors.
 */
template <typename T>
requires std::is_arithmetic_v<T>
class io::ReadEngine : public io::IoBuffer<T> {
  public:
    /** @brief Constructor from file pointer.*/
    ReadEngine(std::FILE * fp) : io::IoBuffer<T>(fp) {}
    /** @brief Invocation operator.
     *  @param dest Pointer to destination data.
     *  @param src Cursor of the starting position to read.
     *  @param size Size in bytes to read, must be a multiple of the size of the template parameter ``T``.
     */
    void operator()(void * dest, const void * src, std::size_t size);
    /** @brief Read data from the current cursor.*/
    void read(T * dest, std::uint64_t count);
    /** @brief Default destructor.*/
    ~ReadEngine(void) = default;
};

// Write engine
// ------------

/** @brief Writing engine.
 *  @details Write an array to a binary file in little endian. The engine also performs checking for writing errors.
 */
template <typename T>
requires std::is_arithmetic_v<T>
class io::WriteEngine : public io::IoBuffer<T> {
  public:
    /** @brief Constructor from file pointer.*/
    WriteEngine(std::FILE * fp) : io::IoBuffer<T>(fp) {}
    /** @brief Invocation operator.
     *  @param dest Pointer to destination data.
     *  @param src Cursor of the starting position to read.
     *  @param size Size in bytes to read, must be a multiple of the size of the template parameter ``T``.
     */
    void operator()(void * dest, const void * src, std::size_t size);
    /** @brief Write data at the current cursor.*/
    void write(const T * src, std::uint64_t count);
    /** @brief Default destructor.*/
    ~WriteEngine(void) = default;
};

}  // namespace merlin

#include "merlin/io/io_engine.tpp"

#endif  // MERLIN_IO_IO_ENGINE_HPP_
