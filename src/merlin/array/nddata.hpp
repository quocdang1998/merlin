// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_NDDATA_HPP_
#define MERLIN_ARRAY_NDDATA_HPP_

#include <cstddef>  // nullptr
#include <cstdint>  // std::uint64_t
#include <string>   // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::NdData
#include "merlin/config.hpp"             // __cuhostdev__, merlin::Index
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/slice.hpp"              // merlin::SliceArray
#include "merlin/vector.hpp"             // merlin::UIntVec

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
    MERLIN_EXPORTS NdData(double * data, const UIntVec & shape, const UIntVec & strides);
    /** @brief Constructor from shape vector.*/
    MERLIN_EXPORTS NdData(const Index & shape);
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
    /** @brief Get reference to shape vector.*/
    __cuhostdev__ constexpr Index & shape(void) noexcept { return this->shape_; }
    /** @brief Get constant reference to shape vector.*/
    __cuhostdev__ constexpr const Index & shape(void) const noexcept { return this->shape_; }
    /** @brief Get reference to stride vector.*/
    __cuhostdev__ constexpr Index & strides(void) noexcept { return this->strides_; }
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
    virtual double get(const Index & index) const { return 0.0; }
    /** @brief Get value of element at a C-contiguous index.*/
    virtual double get(std::uint64_t index) const { return 0.0; }
    /** @brief Set value of element at a n-dim index.*/
    virtual void set(const Index & index, double value) {}
    /** @brief Set value of element at a C-contiguous index.*/
    virtual void set(std::uint64_t index, double value) {}
    /// @}

    /// @name Operations
    /// @{
    /** @brief Reshape the dataset.
     *  @param new_shape New shape.
     */
    MERLIN_EXPORTS void reshape(const UIntVec & new_shape);
    /** @brief Remove dimension with size 1.
     *  @param i_dim Index of dimension to remove.
     */
    MERLIN_EXPORTS void remove_dim(std::uint64_t i_dim);
    /** @brief Collapse all dimensions with size 1.*/
    MERLIN_EXPORTS void squeeze(void);
    /** @brief Set value of all elements.*/
    virtual void fill(double value) {}
    /** @brief Calculate mean and variance of all non-zero and finite elements.*/
    virtual std::array<double, 2> get_mean_variance(void) const { return {0.0, 0.0}; }
    /** @brief Create a sub-array.*/
    virtual array::NdData * sub_array(const SliceArray & slices) const {
        array::NdData * p_result = new array::NdData();
        this->create_sub_array(*p_result, slices);
        return p_result;
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
     *  @details Number of increasing bytes in memory when index of an axis increases by 1.
     */
    Index strides_;

    /// @name Hidden utility functions
    /// @{
    /** @brief Calculate size of array.*/
    MERLIN_EXPORTS void calc_array_size(void) noexcept;
    /** @brief Create sub-array.*/
    MERLIN_EXPORTS void create_sub_array(array::NdData & sub_array, const SliceArray & slices) const noexcept;
    /// @}
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_NDDATA_HPP_
