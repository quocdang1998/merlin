// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_SLICE_HPP_
#define MERLIN_ARRAY_SLICE_HPP_

#include <cstdint>  // std::int64_t, std::uint64_t, INT64_MAX
#include <initializer_list>  // std::initializer_list
#include <tuple>  // std::tuple

#include "merlin/device/decorator.hpp"  // __cuhostdev__

namespace merlin::array {

/** @brief Slice of an Array.*/
class Slice {
  public:
    /// @name Constructors
    /// @{
    /** @brief Member constructor.
     *  @details Construct Slice object from values of its members.
     *  @param start Start position (must be positive).
     *  @param stop Stop position (must be positive).
     *  @param step Step (must be positive).
     */
    __cuhostdev__ Slice(std::uint64_t start = 0, std::int64_t stop = INT64_MAX, std::int64_t step = 1);
    /** @brief Constructor from initializer list.
     *  @param list An initializer list:
     *     - If empty, all elements of the dimension are taken.
     *     - If only 1 element presents, the dimension is collaped.
     *     - If 2 elements, slicing from the first one to the last one.
     *     - If 3 elements, similar to member constructor.
     */
    __cuhostdev__ Slice(std::initializer_list<std::int64_t> list);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ Slice(const Slice & src) : start_(src.start_), stop_(src.stop_), step_(src.step_) {}
    /** @brief Copy assignment.*/
    __cuhostdev__ Slice & operator=(const Slice & src) {
        this->start_ = src.start_;
        this->stop_ = src.stop_;
        this->step_ = src.step_;
        return *this;
    }
    /** @brief Move constructor.*/
    __cuhostdev__ Slice(Slice && src) : start_(src.start_), stop_(src.stop_), step_(src.step_) {}
    /** @brief Move assignment.*/
    __cuhostdev__ Slice & operator=(Slice && src) {
        this->start_ = src.start_;
        this->stop_ = src.stop_;
        this->step_ = src.step_;
        return *this;
    }
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to start value.*/
    __cuhostdev__ std::uint64_t & start(void) {return this->start_;}
    /** @brief Get constant reference to start value.*/
    __cuhostdev__ const std::uint64_t & start(void) const {return this->start_;}
    /** @brief Get reference to stop value.*/
    __cuhostdev__ std::int64_t & stop(void) {return this->stop_;}
    /** @brief Get constant reference to stop value.*/
    __cuhostdev__ const std::int64_t & stop(void) const {return this->stop_;}
    /** @brief Get reference to step value.*/
    __cuhostdev__ std::int64_t & step(void) {return this->step_;}
    /** @brief Get constant reference to step value.*/
    __cuhostdev__ const std::int64_t & step(void) const {return this->step_;}
    /// @}

    /// @name Utilities
    /// @{
    /** @brief Check validity of values inputted in the Slice.*/
    __cuhostdev__ void check_validity(void) const;
    /** @brief Check if Slice indices are in a given range.*/
    __cuhostdev__ bool in_range(std::int64_t lower, std::int64_t upper) const;
    /** @brief Calculate offset, new shape and stride of a dimension.
     *  @param shp Shape of sliced dimension.
     *  @param strd Stride of sliced dimension.
     *  @return A tuple of 3 ``std::uint64_t``, respectively the offset, new shape and new stride of dimension of sliced
     *  array. If the slice has only one element, new shape is 1.
     */
    __cuhostdev__ std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> slice_on(std::uint64_t shp,
                                                                                   std::uint64_t strd) const;
    /// @}

  protected:
    /** @brief Start index.*/
    std::uint64_t start_ = 0;
    /** @brief Stop index, count from last element.*/
    std::int64_t stop_ = INT64_MAX;
    /** @brief Step.*/
    std::int64_t step_ = 1;
};

}  // namespace merlin::array

#endif  // MERLIN_ARRAY_SLICE_HPP_
