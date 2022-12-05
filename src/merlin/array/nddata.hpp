// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_NDDATA_HPP_
#define MERLIN_ARRAY_NDDATA_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::int64_t, std::uint64_t, std::uintptr_t
#include <initializer_list>  // std::initializer_list
#include <tuple>  // std::tie

#include "merlin/array/slice.hpp"  // merlin::array::Slice
#include "merlin/cuda_decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec


namespace merlin::array {
class NdData;  // Basic ndim array
class Array;  // CPU Array, defined in array.hpp
class Parcel;  // GPU Array, defined in parcel.hpp
class Stock;  // Out of core array, defined in stock.hpp
}  // namespace merlin::array


namespace merlin {

/** @brief Abstract class of N-dim array.*/
class MERLIN_EXPORTS array::NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    __cuhostdev__ NdData(void) {}
    /** @brief Constructor from data pointer and meta-data.
     *  @details This constructor is used to construct explicitly an NdData in C++ interface.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Shape vector.
     *  @param strides Strides vector.
     */
    NdData(float * data, std::uint64_t ndim, const intvec & shape, const intvec & strides);
    /** @brief Constructor from data pointer and meta-data pointers.
     *  @details This constructor is designed for initializing object from Numpy np.array.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Pointer to shape vector.
     *  @param strides Pointer to strides vector.
     */
    NdData(float * data, std::uint64_t ndim, const std::uint64_t * shape, const std::uint64_t * strides);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::NdData of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    __cuhostdev__ NdData(const array::NdData & whole, std::initializer_list<array::Slice> slices);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Shallow copy constructor.*/
    NdData(const array::NdData & source) = default;
    /** @brief Shallow copy assignment.*/
    array::NdData & operator= (const array::NdData & source) = default;
    /** @brief Move constructor.*/
    NdData(array::NdData && source) = default;
    /** @brief Move assignment.*/
    array::NdData & operator= (array::NdData && source) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get pointer to data.*/
    __cuhostdev__ float * data(void) const {return this->data_;}
    /** @brief Get number of dimension.*/
    __cuhostdev__ std::uint64_t ndim(void) const {return this->ndim_;}
    /** @brief Get reference to shape vector.*/
    __cuhostdev__ intvec & shape(void) {return this->shape_;}
    /** @brief Get constant reference to shape vector.*/
    __cuhostdev__ const intvec & shape(void) const {return this->shape_;}
    /** @brief Get reference to stride vector.*/
    __cuhostdev__ intvec & strides(void) {return this->strides_;}
    /** @brief Get constant reference to stride vector.*/
    __cuhostdev__ const intvec & strides(void) const {return this->strides_;}
    /// @}

    /// @name Atributes
    /// @{
    /** @brief Number of element.*/
    __cuhostdev__ std::uint64_t size(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    __cuhostdev__ ~NdData(void) {}
    /// @}

  protected:
    /** @brief Pointer to data.*/
    float * data_ = nullptr;
    /** @brief Number of dimension.*/
    std::uint64_t ndim_;
    /** @brief Shape vector.
     *  @details Size of each dimension.
     */
    intvec shape_;
    /** @brief Stride vector.
     *  @details Number of incresing bytes in memory when an intvec of a dimension jumps by 1.
     */
    intvec strides_;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_NDDATA_HPP_
