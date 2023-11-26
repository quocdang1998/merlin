// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_ARRAY_HPP_
#define MERLIN_ARRAY_ARRAY_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/array/nddata.hpp"       // merlin::array::NdData
#include "merlin/cuda/stream.hpp"        // merlin::cuda::Stream
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/iterator.hpp"           // merlin::Iterator
#include "merlin/shuffle.hpp"            // merlin::Shuffle
#include "merlin/slice.hpp"              // merlin::slicevec
#include "merlin/vector.hpp"             // merlin::intvec

namespace merlin {

// Allocate non-pageable memory
// ----------------------------

namespace array {

/** @brief Allocate non pageable memory.
 *  @param size Number of element in the allocated array.
 */
double * allocate_memory(std::uint64_t size);

/** @brief Pin memory to RAM.*/
void cuda_pin_memory(double * ptr, std::uint64_t n_elem);

/** @brief Free array allocated in non pageable memory.*/
void free_memory(double * ptr);

}  // namespace array

// Array class
// -----------

/** @brief Multi-dimensional array on CPU.*/
class array::Array : public array::NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    Array(void) = default;
    /** @brief Construct 1D array holding a double precision value.
     *  @param value Assigned value.
     */
    MERLIN_EXPORTS Array(double value);
    /** @brief Construct C-contiguous empty array from dimension vector.
     *  @param shape Shape vector.
     */
    MERLIN_EXPORTS Array(const intvec & shape);
    /** @brief Construct array from pointer, to data and meta-data.
     *  @param data Pointer to data.
     *  @param shape Size per dimension.
     *  @param strides Stride per dimension.
     *  @param copy Copy the original tensor to C-contiguous tensor.
     *  @note The original memory tied to the pointer will not be freed at destruction. But if copy is true, the
        copied tensor is automatically deallocated inside the destructor.
     */
    MERLIN_EXPORTS Array(double * data, const intvec & shape, const intvec & strides, bool copy = false);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::Array of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    MERLIN_EXPORTS Array(const array::Array & whole, const slicevec & slices);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Deep copy constructor.*/
    MERLIN_EXPORTS Array(const array::Array & src);
    /** @brief Deep copy assignment.*/
    MERLIN_EXPORTS Array & operator=(const array::Array & src);
    /** @brief Move constructor.*/
    MERLIN_EXPORTS Array(array::Array && src);
    /** @brief Move assignment.*/
    MERLIN_EXPORTS Array & operator=(array::Array && src);
    /// @}

    /// @name Slicing operator
    /// @{
    /** @brief Get reference to element at a given ndim index.*/
    MERLIN_EXPORTS double & operator[](const intvec & index);
    /** @brief Get reference to element at a given C-contiguous index.*/
    MERLIN_EXPORTS double & operator[](std::uint64_t index);
    /** @brief Get constant reference to element at a given ndim index.*/
    MERLIN_EXPORTS const double & operator[](const intvec & index) const;
    /** @brief Get const reference to element at a given C-contiguous index.*/
    MERLIN_EXPORTS const double & operator[](std::uint64_t index) const;
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    MERLIN_EXPORTS double get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    MERLIN_EXPORTS double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    MERLIN_EXPORTS void set(const intvec index, double value);
    /** @brief Set value of element at a C-contiguous index.*/
    MERLIN_EXPORTS void set(std::uint64_t index, double value);
    /// @}

    /// @name Operations
    /// @{
    /** @brief Set value of all elements.*/
    MERLIN_EXPORTS void fill(double value);
    /// @}

    /// @name Transfer data
    /// @{
    /** @brief Copy data from GPU array.*/
    MERLIN_EXPORTS void clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream = cuda::Stream());
    /** @brief Export data to a file.
     *  @param src Exported array.
     */
    MERLIN_EXPORTS void extract_data_from_file(const array::Stock & src);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(bool first_call = true) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Array(void);
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_ARRAY_HPP_
