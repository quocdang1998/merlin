// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_ARRAY_HPP_
#define MERLIN_ARRAY_ARRAY_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <string>  // std::string

#include "merlin/array/nddata.hpp"  // merlin::array::NdData, merlin::array::Parcel
#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/iterator.hpp"  // merlin::Iterator
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

/** @brief Allocate non pageable memory.
 *  @param size Number of element in the allocated array.
 */
float * allocate_memory(std::uint64_t size);

/** @brief Free array allocated in non pageable memory.*/
void free_memory(float * ptr, std::uint64_t size);

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
    Array(const array::Array & whole, const Vector<array::Slice> & slices);
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

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    float get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    float get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    void set(const intvec index, float value);
    /** @brief Set value of element at a C-contiguous index.*/
    void set(std::uint64_t index, float value);
    /// @}

    /// @name Transfer data
    /// @{
    /** @brief Copy data from GPU array.*/
    void sync_from_gpu(const array::Parcel & gpu_array, const cuda::Stream & stream = cuda::Stream());
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
    /** @brief Decision to delete merlin::array::Array::data_ at destruction or not.*/
    bool force_free;
    /** @brief Index vector of begin element.*/
    intvec begin_;
    /** @brief Index vector of last element.*/
    intvec end_;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_ARRAY_HPP_
