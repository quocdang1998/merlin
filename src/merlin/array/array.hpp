// Copyright 2022 quocdang1998NdData
#ifndef MERLIN_ARRAY_ARRAY_HPP_
#define MERLIN_ARRAY_ARRAY_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <initializer_list>  // std::initializer_list

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/array/nddata.hpp"  // merlin::NdData, merlin::Parcel, merlin::Iterator
#include "merlin/array/slice.hpp"  // merlin::Slice
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Multi-dimensional array on CPU.*/
class MERLIN_EXPORTS Array : public NdData {
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
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Deep copy constructor.*/
    Array(const Array & src);
    /** @brief Deep copy assignment.*/
    Array & operator=(const Array & src);
    /** @brief Move constructor.*/
    Array(Array && src);
    /** @brief Move assignment.*/
    Array & operator=(Array && src);
    /// @}

    /// @name Iterator
    /// @{
    /** @brief Iterator class.*/
    using iterator = Iterator;
    /** @brief Begin iterator.
     *  @details Vector of index \f$(0, 0, ..., 0)\f$.
     */
    Array::iterator begin(void);
    /** @brief End iterator.
     *  @details Vector of index \f$(d_0, 0, ..., 0)\f$.
     */
    Array::iterator end(void);
    /** @brief Sciling operator.
     *  @details Get an element at a given index.
     *  @param index Vector of indices along each dimension.
     *  @return Reference to the element at the provided index.
     */
    float & operator[] (const intvec & index);
    /** @brief Create new array by slicing.*/
    // Array operator[] (std::initializer_list<Slice> slices);
    /// @}

    /// @name Transfer data
    /// @{
    /** @brief Copy data from GPU array.*/
    void sync_from_gpu(const Parcel & gpu_array, std::uintptr_t stream = 0);
    /** @brief Export data to a file.
     *  @param filename Name of exported file.
     */
    void export_to_file(const std::string & filename);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Array(void);
    /// @}

  protected:
    // Members
    // -------
    /** @brief Decision to delete Array::data_ at destruction or not.*/
    bool force_free;
    /** @brief Index vector of begin element.*/
    intvec begin_;
    /** @brief Index vector of last element.*/
    intvec end_;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_ARRAY_HPP_
