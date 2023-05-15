// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_SLICE_HPP_
#define MERLIN_ARRAY_SLICE_HPP_

#include <array>  // std::array
#include <cstdint>  // std::uint64_t, UINT64_MAX
#include <initializer_list>  // std::initializer_list
#include <string>  // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Slice
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__

namespace merlin {

/** @brief %Slice of an array.*/
class array::Slice {
  public:
    /// @name Constructors
    /// @{
    /** @brief Member constructor.
     *  @details Construct object from values of its members.
     *  @param start Start position.
     *  @param stop Stop position.
     *  @param step Step.
     */
    __cuhostdev__ Slice(std::uint64_t start = 0, std::uint64_t stop = UINT64_MAX, std::uint64_t step = 1);
    /** @brief Constructor from initializer list.
     *  @param list An initializer list:
     *     - If empty, all elements of the dimension are taken.
     *     - If only 1 element presents, the dimension is collaped.
     *     - If 2 elements, slicing from the first one to the last one.
     *     - If 3 elements, similar to member constructor.
     */
    __cuhostdev__ Slice(std::initializer_list<std::uint64_t> list);
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
    __cuhostdev__ constexpr std::uint64_t & start(void) noexcept {return this->start_;}
    /** @brief Get constant reference to start value.*/
    __cuhostdev__ constexpr const std::uint64_t & start(void) const noexcept {return this->start_;}
    /** @brief Get reference to stop value.*/
    __cuhostdev__ constexpr std::uint64_t & stop(void) noexcept {return this->stop_;}
    /** @brief Get constant reference to stop value.*/
    __cuhostdev__ constexpr const std::uint64_t & stop(void) const noexcept {return this->stop_;}
    /** @brief Get reference to step value.*/
    __cuhostdev__ constexpr std::uint64_t & step(void) noexcept {return this->step_;}
    /** @brief Get constant reference to step value.*/
    __cuhostdev__ constexpr const std::uint64_t & step(void) const noexcept {return this->step_;}
    /// @}

    /// @name Utilities
    /// @{
    /** @brief Calculate offset, new shape and stride of a dimension.
     *  @param shape Shape of sliced dimension.
     *  @param stride Stride of sliced dimension.
     *  @return A tuple of 3 ``std::uint64_t``, respectively the offset, new shape and new stride of dimension of
     *  sliced array. If the slice has only one element, new shape is 1.
     */
    __cuhostdev__ std::array<std::uint64_t, 3> slice_on(std::uint64_t shape, std::uint64_t stride) const;
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
    __cuhostdev__ ~Slice(void);
    /// @}

  protected:
    /** @brief Start index.*/
    std::uint64_t start_ = 0;
    /** @brief Stop index, count from last element.*/
    std::uint64_t stop_ = UINT64_MAX;
    /** @brief Step.*/
    std::uint64_t step_ = 1;

  private:
    /** @brief Check validity of values inputted in the Slice.*/
    __cuhostdev__ void check_validity(void) const;
};

}  // namespace merlin

#endif  // MERLIN_ARRAY_SLICE_HPP_
