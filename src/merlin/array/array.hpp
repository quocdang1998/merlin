// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_ARRAY_HPP_
#define MERLIN_ARRAY_ARRAY_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <string>  // std::string
#include <initializer_list>  // std::initializer_list

#include "merlin/array/nddata.hpp"  // merlin::array::NdData, merlin::array::Parcel
#include "merlin/array/slice.hpp"  // merlin::Slice
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Multi-dimensional array on CPU.*/
class MERLIN_EXPORTS array::Array : public array::NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    Array(void) = default;
    /** @brief Construct 1D array holding a float value.
     *  @param value Assigned value.
     */
    Array(float value);
    /** @brief Construct C-contiguous empty array from dimension vector.
     *  @param shape Shape vector.
     */
    Array(const intvec & shape);
    /** @brief Construct array from pointer, to data and meta-data.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension of tensor.
     *  @param shape Pointer to tensor to size per dimension.
     *  @param strides Pointer to tensor to stride per dimension.
     *  @param copy Copy the original tensor to C-contiguous tensor.
     *  @note The original memory tied to the pointer will not be freed at destruction. But if copy is true, the
        copied tensor is automatically deallocated inside the destructor.
     */
    Array(float * data, std::uint64_t ndim,
          const std::uint64_t * shape, const std::uint64_t * strides, bool copy = false);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::NdData of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    Array(const array::Array & whole, std::initializer_list<array::Slice> slices);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Deep copy constructor.*/
    Array(const array::Array & src);
    /** @brief Deep copy assignment.*/
    Array & operator=(const array::Array & src);
    /** @brief Move constructor.*/
    Array(array::Array && src);
    /** @brief Move assignment.*/
    Array & operator=(array::Array && src);
    /// @}

    /// @name Iterator
    /// @{
    /** @brief Iterator class.*/
    using iterator = Iterator;
    /** @brief Begin iterator.
     *  @details Vector of index \f$(0, 0, ..., 0)\f$.
     */
    array::Array::iterator begin(void);
    /** @brief End iterator.
     *  @details Vector of index \f$(d_0, 0, ..., 0)\f$.
     */
    array::Array::iterator end(void);
    /** @brief Sciling operator.
     *  @details Get an element at a given index.
     *  @param index Vector of indices along each dimension.
     *  @return Reference to the element at the provided index.
     */
    float & operator[](const intvec & index);
    /// @}

    /// @name Transfer data
    /// @{
    /** @brief Copy data from GPU array.*/
    void sync_from_gpu(const array::Parcel & gpu_array, std::uintptr_t stream = 0);
    /** @brief Export data to a file.
     *  @param filename Name of exported file.
     */
    // void export_to_file(const std::string & filename);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Array(void);
    /// @}

  protected:
    /** @brief Decision to delete merlin::array::Array::data_ at destruction or not.*/
    bool force_free;
    /** @brief Index vector of begin element.*/
    intvec begin_;
    /** @brief Index vector of last element.*/
    intvec end_;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_ARRAY_HPP_
