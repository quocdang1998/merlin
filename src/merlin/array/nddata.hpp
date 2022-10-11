// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_NDDATA_HPP_
#define MERLIN_ARRAY_NDDATA_HPP_

#include <cstddef>  // NULL
#include <cstdint>  // std::int64_t, std::uint64_t, std::uintptr_t
#include <tuple>  // std::tie

#include "merlin/device/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin::array {

/** @brief Abstract class of N-dim array.*/
class MERLIN_EXPORTS NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    NdData(void) = default;
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
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Shallow copy constructor.*/
    NdData(const NdData & source) = default;
    /** @brief Shallow copy assignment.*/
    NdData & operator= (const NdData & source) = default;
    /** @brief Move constructor.*/
    NdData(NdData && source) = default;
    /** @brief Move assignment.*/
    NdData & operator= (NdData && source) = default;
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
    std::uint64_t size(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    virtual ~NdData(void) = default;
    /// @}


  protected:
    /** @brief Pointer to data.*/
    float * data_ = NULL;
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

// Forward declaration of subclasses
class Array;  // CPU Array, defined in tensor.hpp
class Parcel;  // GPU Array, defined in parcel.hpp

}  // namespace merlin::array

#endif  // MERLIN_ARRAY_NDDATA_HPP_
