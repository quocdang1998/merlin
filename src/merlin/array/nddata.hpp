// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_NDDATA_HPP_
#define MERLIN_ARRAY_NDDATA_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::int64_t, std::uint64_t, std::uintptr_t
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::NdData
#include "merlin/cuda_interface.hpp"     // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/ndindex.hpp"            // merlin::Index
#include "merlin/slice.hpp"              // merlin::slicevec
#include "merlin/vector.hpp"             // merlin::intvec

namespace merlin {

/** @brief Abstract class of N-dim array.*/
class array::NdData {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor (do nothing).*/
    NdData(void) = default;
    /** @brief Constructor from data pointer and meta-data.
     *  @details This constructor is used to construct explicitly an NdData in C++ interface.
     *  @param data Pointer to data.
     *  @param shape Shape vector.
     *  @param strides Strides vector.
     */
    MERLIN_EXPORTS NdData(double * data, const intvec & shape, const intvec & strides);
    /** @brief Constructor from shape vector.*/
    MERLIN_EXPORTS NdData(const intvec & shape);
    /** @brief Constructor from a slice.
     *  @param whole merlin::array::NdData of the original array.
     *  @param slices List of merlin::array::Slice on each dimension.
     */
    // MERLIN_EXPORTS NdData(const array::NdData & whole, const slicevec & slices);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Shallow copy constructor.*/
    NdData(const array::NdData & source) = default;
    /** @brief Shallow copy assignment.*/
    array::NdData & operator=(const array::NdData & source) = default;
    /** @brief Move constructor.*/
    NdData(array::NdData && source) = default;
    /** @brief Move assignment.*/
    array::NdData & operator=(array::NdData && source) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get pointer to data.*/
    __cuhostdev__ constexpr double * data(void) const noexcept { return this->data_; }
    /** @brief Get number of dimension.*/
    __cuhostdev__ constexpr const std::uint64_t & ndim(void) const noexcept { return this->ndim_; }
    /** @brief Get constant reference to shape vector.*/
    __cuhostdev__ constexpr const Index & shape(void) const noexcept { return this->shape_; }
    /** @brief Get constant reference to stride vector.*/
    __cuhostdev__ constexpr const Index & strides(void) const noexcept { return this->strides_; }
    /// @}

    /// @name Atributes
    /// @{
    /** @brief Get number of element.*/
    __cuhostdev__ constexpr const std::uint64_t & size(void) const noexcept { return this->size_; }
    /** @brief Check if the array is C-contiguous.*/
    MERLIN_EXPORTS bool is_c_contiguous(void) const;
    /** @brief Release memory in destructor.*/
    bool release = false;
    /// @}

    /// @name Get and set element
    /// @{
    /** @brief Get value of element at a n-dim index.*/
    virtual double get(const intvec & index) const { return 0.0; }
    /** @brief Get value of element at a C-contiguous index.*/
    virtual double get(std::uint64_t index) const { return 0.0; }
    /** @brief Set value of element at a n-dim index.*/
    virtual void set(const intvec index, double value) {}
    /** @brief Set value of element at a C-contiguous index.*/
    virtual void set(std::uint64_t index, double value) {}
    /// @}

    /// @name Operations
    /// @{
    /** @brief Reshape the dataset.
     *  @param new_shape New shape.
     */
    MERLIN_EXPORTS void reshape(const intvec & new_shape);
    /** @brief Remove dimension with size 1.
     *  @param i_dim Index of dimension to remove.
     */
    MERLIN_EXPORTS void remove_dim(std::uint64_t i_dim);
    /** @brief Collapse all dimensions with size 1.*/
    MERLIN_EXPORTS void squeeze(void);
    /** @brief Set value of all elements.*/
    virtual void fill(double value) {}
    /** @brief Create a sub-array.*/
    array::NdData sub_array(const slicevec & slices) const {
        array::NdData result;
        this->create_sub_array(result, slices);
        return result;
    }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS virtual std::string str(bool first_call = true) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS virtual ~NdData(void);
    /// @}

  protected:
    /** @brief Pointer to data.*/
    double * data_ = nullptr;
    /** @brief Number of elements.*/
    std::uint64_t size_ = 0;
    /** @brief Number of dimensions.*/
    std::uint64_t ndim_ = 0;
    /** @brief Shape vector.
     *  @details Size of each axis.
     */
    Index shape_;
    /** @brief Stride vector.
     *  @details Number of increasing bytes in memory when an intvec of a dimension jumps by 1.
     */
    Index strides_;

    /** @brief Calculate size of array.*/
    MERLIN_EXPORTS void calc_array_size(void) noexcept;
    /** @brief Create sub-array.*/
    MERLIN_EXPORTS void create_sub_array(array::NdData & sub_array, const slicevec & slices) const noexcept;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_NDDATA_HPP_
