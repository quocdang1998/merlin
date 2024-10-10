// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_ARRAY_HPP_
#define MERLIN_ARRAY_ARRAY_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel, merlin::array::Stock
#include "merlin/array/nddata.hpp"       // merlin::array::NdData
#include "merlin/cuda/stream.hpp"        // merlin::cuda::Stream
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::Index

namespace merlin {

// Array class
// -----------

/** @brief Multi-dimensional array on CPU.*/
class array::Array : public array::NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    Array(void) = default;
    /** @brief Construct array from pointer, to data and meta-data.
     *  @param data Pointer to data.
     *  @param shape Size per dimension.
     *  @param strides Stride per dimension.
     *  @param copy Copy the original tensor to C-contiguous tensor.
     *  @param pin_memory If compiled with CUDA, pin pages containing assigned data to the memory. Pinned memory pages
     *  cannot be swapped out of the memory. Memory should only be pages once, and the paging order must be performed on
     *  the assigned array containing the first element of the whole array.
     *  @note The original memory tied to the pointer will not be freed at destruction. But if copy is true, the
        copied tensor is automatically deallocated inside the destructor.
     */
    MERLIN_EXPORTS Array(double * data, const Index & shape, const Index & strides, bool copy = false,
                         bool pin_memory = true);
    /** @brief Constructor C-contiguous empty array from shape vector.*/
    MERLIN_EXPORTS Array(const Index & shape);
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
    MERLIN_EXPORTS double & operator[](const Index & index);
    /** @brief Get reference to element at a given C-contiguous index.*/
    MERLIN_EXPORTS double & operator[](std::uint64_t index);
    /** @brief Get constant reference to element at a given ndim index.*/
    MERLIN_EXPORTS const double & operator[](const Index & index) const;
    /** @brief Get const reference to element at a given C-contiguous index.*/
    MERLIN_EXPORTS const double & operator[](std::uint64_t index) const;
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    MERLIN_EXPORTS double get(const Index & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    MERLIN_EXPORTS double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    MERLIN_EXPORTS void set(const Index & index, double value);
    /** @brief Set value of element at a C-contiguous index.*/
    MERLIN_EXPORTS void set(std::uint64_t index, double value);
    /// @}

    /// @name Operations
    /// @{
    /** @brief Set value of all elements.*/
    MERLIN_EXPORTS void fill(double value);
    /** @brief Calculate mean and variance of all non-zero and finite elements.*/
    MERLIN_EXPORTS std::array<double, 2> get_mean_variance(void) const;
    /** @brief Create a polymorphic sub-array.*/
    array::NdData * get_p_sub_array(const SliceArray & slices) const {
        array::Array * p_result = new array::Array();
        this->create_sub_array(*p_result, slices);
        return p_result;
    }
    /** @brief Create sub-array with the same type.*/
    array::Array get_sub_array(const SliceArray & slices) const {
        array::Array sub_array;
        this->create_sub_array(sub_array, slices);
        return sub_array;
    }
    /// @}

    /// @name Transfer data
    /// @{
    /** @brief Copy data from GPU array.
     *  @warning This function will lock the mutex.
     */
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

  protected:
    /** @brief Flag indicating if the memory page is pinned to RAM.*/
    bool is_pinned = false;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_ARRAY_HPP_
