// Copyright 2022 quocdang1998
#ifndef MERLIN_SLICE_HPP_
#define MERLIN_SLICE_HPP_

#include <array>             // std::array
#include <cmath>             // std::ceil
#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <limits>            // std::numeric_limits
#include <string>            // std::string

#include "merlin/config.hpp"                // __cuhostdev__, merlin::max_dim
#include "merlin/exports.hpp"               // MERLIN_EXPORTS
#include "merlin/vector/static_vector.hpp"  // merlin::vector::StaticVector

namespace merlin {

/** @brief %Slice of an array.*/
class Slice {
  public:
    /// @name Constructors
    /// @{
    /** @brief Member constructor.
     *  @details Construct object from values of its members.
     *  @param start Start position.
     *  @param stop Stop position.
     *  @param step Step.
     */
    Slice(std::uint64_t start = 0, std::uint64_t stop = std::numeric_limits<std::uint64_t>::max(),
          std::uint64_t step = 1) :
    start_(start), stop_(stop), step_(step) {
        this->check_validity();
    }
    /** @brief Constructor from initializer list.
     *  @param list An initializer list:
     *     - If empty, all elements of the dimension are taken.
     *     - If only 1 element presents, the dimension is collaped.
     *     - If 2 elements, slicing from the first one to the last one.
     *     - If 3 elements, similar to member constructor.
     */
    MERLIN_EXPORTS Slice(std::initializer_list<std::uint64_t> list);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Slice(const Slice & src) : start_(src.start_), stop_(src.stop_), step_(src.step_) {}
    /** @brief Copy assignment.*/
    Slice & operator=(const Slice & src) {
        this->start_ = src.start_;
        this->stop_ = src.stop_;
        this->step_ = src.step_;
        return *this;
    }
    /** @brief Move constructor.*/
    Slice(Slice && src) : start_(src.start_), stop_(src.stop_), step_(src.step_) {}
    /** @brief Move assignment.*/
    Slice & operator=(Slice && src) {
        this->start_ = src.start_;
        this->stop_ = src.stop_;
        this->step_ = src.step_;
        return *this;
    }
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to start value.*/
    __cuhostdev__ constexpr std::uint64_t & start(void) noexcept { return this->start_; }
    /** @brief Get constant reference to start value.*/
    __cuhostdev__ constexpr const std::uint64_t & start(void) const noexcept { return this->start_; }
    /** @brief Get reference to stop value.*/
    __cuhostdev__ constexpr std::uint64_t & stop(void) noexcept { return this->stop_; }
    /** @brief Get constant reference to stop value.*/
    __cuhostdev__ constexpr const std::uint64_t & stop(void) const noexcept { return this->stop_; }
    /** @brief Get reference to step value.*/
    __cuhostdev__ constexpr std::uint64_t & step(void) noexcept { return this->step_; }
    /** @brief Get constant reference to step value.*/
    __cuhostdev__ constexpr const std::uint64_t & step(void) const noexcept { return this->step_; }
    /// @}

    /// @name Utilities
    /// @{
    /** @brief Calculate offset, new shape and stride of a dimension.
     *  @param shape Shape of sliced dimension.
     *  @param stride Stride of sliced dimension.
     *  @return A tuple of 3 ``std::uint64_t``, respectively the offset, new shape and new stride of dimension of
     *  sliced array. If the slice has only one element, new shape is 1.
     */
    __cuhostdev__ std::array<std::uint64_t, 3> slice_on(std::uint64_t shape, std::uint64_t stride) const {
        std::uint64_t stop = (this->stop_ > shape) ? shape : this->stop_;
        double steps = static_cast<double>(stop - this->start_) / static_cast<double>(this->step_);
        std::uint64_t new_shp = (steps < 0) ? 0 : std::uint64_t(std::ceil(steps));
        std::uint64_t offset = this->start_ * stride;
        std::uint64_t new_strd = this->step_ * stride;
        return {offset, new_shp, new_strd};
    }
    /** @brief Get index in whole array given index in a sliced array.*/
    __cuhostdev__ constexpr std::uint64_t get_index_in_whole_array(std::uint64_t index_sliced_array) const noexcept {
        return this->start_ + (this->step_ * index_sliced_array);
    }
    /** @brief Check if a given index wrt. the full array is in the slice.*/
    __cuhostdev__ constexpr bool in_slice(std::uint64_t index) const noexcept {
        if ((index < this->start_) || (index >= this->stop_)) {
            return false;
        }
        return ((index - this->start_) % this->step_) == 0;
    }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~Slice(void) = default;
    /// @}

  protected:
    /** @brief Start index.*/
    std::uint64_t start_ = 0;
    /** @brief Stop index, count from last element.*/
    std::uint64_t stop_ = std::numeric_limits<std::uint64_t>::max();
    /** @brief Step.*/
    std::uint64_t step_ = 1;

  private:
    /** @brief Check validity of values inputted in the Slice.*/
    MERLIN_EXPORTS void check_validity(void) const;
};

/** @brief Array of slices.*/
using SliceArray = vector::StaticVector<Slice, max_dim>;

}  // namespace merlin

#endif  // MERLIN_SLICE_HPP_
