// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_NDDATA_HPP_
#define MERLIN_ARRAY_NDDATA_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::int64_t, std::uint64_t, std::uintptr_t
#include <utility>  // std::pair

#include "merlin/array/declaration.hpp"  // merlin::array::NdData, merlin::array::Slice
#include "merlin/cuda_decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec


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
    NdData(double * data, std::uint64_t ndim, const intvec & shape, const intvec & strides);
    /** @brief Constructor from data pointer and meta-data pointers.
     *  @details This constructor is designed for initializing object from Numpy np.array.
     *  @param data Pointer to data.
     *  @param ndim Number of dimension.
     *  @param shape Pointer to shape vector.
     *  @param strides Pointer to strides vector.
     */
    NdData(double * data, std::uint64_t ndim, const std::uint64_t * shape, const std::uint64_t * strides);
    /** @brief Constructor from shape vector.*/
    NdData(const intvec & shape);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::NdData of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    __cuhostdev__ NdData(const array::NdData & whole, const Vector<array::Slice> & slices);
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
    __cuhostdev__ constexpr double * data(void) const noexcept {return this->data_;}
    /** @brief Get number of dimension.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept {return this->ndim_;}
    /** @brief Get constant reference to shape vector.*/
    __cuhostdev__ constexpr const intvec & shape(void) const noexcept {return this->shape_;}
    /** @brief Get constant reference to stride vector.*/
    __cuhostdev__ constexpr const intvec & strides(void) const noexcept {return this->strides_;}
    /// @}

    /// @name Atributes
    /// @{
    /** @brief Number of element.*/
    __cuhostdev__ std::uint64_t size(void) const noexcept;
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    virtual double get(const intvec & index) const;
    /** @brief Get value of element at a C-contiguous index.*/
    virtual double get(std::uint64_t index) const;
    /** @brief Set value of element at a n-dim index.*/
    virtual void set(const intvec index, double value);
    /** @brief Set value of element at a C-contiguous index.*/
    virtual void set(std::uint64_t index, double value);
    /// @}

    /// @name Partite data
    /// @{
    /** @brief Partite a big array into smaller array given a limit size to each subsidary array.
     *  @param max_memory Limit size of each subsidary array.
     *  @return A tuple of limit dimension and number of sub-array. If the original array fits in the memory, a tuple
     *  of ``UINT64_MAX, UINT64_MAX`` is returned.
     */
    Vector<Vector<array::Slice>> partite(std::uint64_t max_memory);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    virtual ~NdData(void);
    /// @}

  protected:
    /** @brief Pointer to data.*/
    double * data_ = nullptr;
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
    /** @brief Release memory in destructor.*/
    bool release_ = false;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_NDDATA_HPP_
