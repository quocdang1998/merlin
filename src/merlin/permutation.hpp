// Copyright 2023 quocdang1998
#ifndef MERLIN_PERMUTATION_HPP_
#define MERLIN_PERMUTATION_HPP_

#include <iterator>  // std::random_access_iterator
#include <string>    // std::string
#include <utility>   // std::swap

#include "merlin/config.hpp"   // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"   // merlin::IntView, merlin::IntVec, merlin::UIntVec

namespace merlin {

/** @brief %Permutation operation.*/
class Permutation {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    Permutation(void) = default;
    /** @brief Constructor of an identity permutation given its range.*/
    MERLIN_EXPORTS Permutation(std::uint64_t range);
    /** @brief Constructor from permutation index.*/
    MERLIN_EXPORTS Permutation(const UIntVec & index);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Permutation(const Permutation & src) = default;
    /** @brief Copy assignment.*/
    Permutation & operator=(const Permutation & src) = default;
    /** @brief Move constructor.*/
    Permutation(Permutation && src) = default;
    /** @brief Move assignment.*/
    Permutation & operator=(Permutation && src) = default;
    /// @}

    /// @name Attributes
    /// @{
    /** @brief Constant reference to index vector.*/
    __cuhostdev__ constexpr IntView index(void) const noexcept { return this->index_.get_view(); }
    /** @brief Get number of elements permuted.*/
    __cuhostdev__ constexpr std::uint64_t size(void) const noexcept { return this->index_.size(); }
    /// @}

    /// @name Modification
    /// @{
    /** @brief Permutate 2 indices.*/
    __cuhostdev__ constexpr void transpose(std::uint64_t index1, std::uint64_t index2) noexcept {
        std::swap(this->index_[index1], this->index_[index2]);
    }
    /// @}

    /// @name Permutate index of a range
    /// @{
    /** @brief Permute and copy a range into another.
     *  @tparam InputIterator Iterator of the source container.
     *  @tparam OutputIterator Iterator of the destination container.
     *  @param src Iterator to the first element of the source container.
     *  @param dest Iterator to the first element of the destination container.
     */
    template <typename InputIterator, typename OutputIterator>
    requires std::random_access_iterator<InputIterator> && std::random_access_iterator<OutputIterator>
    __cuhostdev__ constexpr void permute(InputIterator src, OutputIterator dest) const {
        for (std::uint64_t i = 0; i < this->size(); i++) {
            *(dest + this->index_[i]) = *(src + i);
        }
    }
    /** @brief Permute a range inplace without copying.
     *  @tparam Iterator Iterator of the array container.
     *  @param dest Iterator to the first element of the array container to perform the permutation.
     */
    template <typename Iterator>
    requires std::random_access_iterator<Iterator>
    __cuhostdev__ constexpr void inplace_permute(Iterator dest) {
        for (std::int64_t i = 0; i < this->size(); i++) {
            // skip to the next non-processed item
            if (this->index_[i] < 0) {
                continue;
            }
            // swapping elemental cycles for non-processed element
            std::int64_t current_position = static_cast<std::int64_t>(i);
            std::int64_t target = this->index_[current_position];
            while (target != i) {
                target = this->index_[current_position];
                std::swap(*(dest + i), *(dest + target));
                this->index_[current_position] = -1 - this->index_[current_position];
                current_position = target;
            }
            // mark last current position as swapped before moving on
            this->index_[current_position] = -1 - this->index_[current_position];
        }
        // reset the index array back to original value
        for (std::uint64_t i = 0; i < this->size(); i++) {
            this->index_[i] = -1 - this->index_[i];
        }
    }
    /// @}

    /// @name Inversion
    /// @{
    /** @brief Calculate the inverse permutation.*/
    MERLIN_EXPORTS Permutation inv(void) const;
    /** @brief Inverse the permutation of a range and copy into another.
     *  @tparam InputIterator Iterator of the source container.
     *  @tparam OutputIterator Iterator of the destination container.
     *  @param src Iterator to the first element of the source container.
     *  @param dest Iterator to the first element of the destination container.
     */
    template <typename InputIterator, typename OutputIterator>
    requires std::random_access_iterator<InputIterator> && std::random_access_iterator<OutputIterator>
    __cuhostdev__ constexpr void inv_permute(InputIterator src, OutputIterator dest) const {
        for (std::uint64_t i = 0; i < this->size(); i++) {
            *(dest + i) = *(src + this->index_[i]);
        }
    }
    /** @brief Inverse the permutation of a range inplace without copying.
     *  @tparam Iterator Iterator of the array container.
     *  @param dest Iterator to the first element of the array container to perform the permutation.
     */
    template <typename Iterator>
    requires std::random_access_iterator<Iterator>
    __cuhostdev__ constexpr void inplace_inv_permute(Iterator dest) {
        for (std::uint64_t i = 0; i < this->size(); i++) {
            // skip to the next non-processed item
            if (this->index_[i] < 0) {
                continue;
            }
            // swapping elemental cycles for non-processed element
            std::int64_t current_position = static_cast<std::int64_t>(i);
            while (this->index_[current_position] != i) {
                const std::int64_t target = this->index_[current_position];
                std::swap(*(dest + current_position), *(dest + target));
                this->index_[current_position] = -1 - this->index_[current_position];
                current_position = target;
            }
            // mark last current position as swapped before moving on
            this->index_[current_position] = -1 - this->index_[current_position];
        }
        // reset the index array back to original value
        for (std::uint64_t i = 0; i < this->size(); i++) {
            this->index_[i] = -1 - this->index_[i];
        }
    }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~Permutation(void) = default;
    /// @}

  protected:
    /** @brief Permutation index.*/
    IntVec index_;
};

}  // namespace merlin

#endif  // MERLIN_PERMUTATION_HPP_
