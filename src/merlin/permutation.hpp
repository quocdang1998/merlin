// Copyright 2023 quocdang1998
#ifndef MERLIN_PERMUTATION_HPP_
#define MERLIN_PERMUTATION_HPP_

#include <utility>  // std::swap
#include <string>  // std::string

#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"   // merlin::UIntVec

namespace merlin {

/** @brief %Permutation operation.*/
class Permutation {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    Permutation(void) = default;
    /** @brief Constructor of a random permutation given its range.*/
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
    __cuhostdev__ constexpr const Vector<std::int64_t> & index(void) const noexcept { return this->index_; }
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
    __cuhostdev__ constexpr void permute(InputIterator src, OutputIterator dest) const {
        for (std::uint64_t i = 0; i < this->size(); i++) {
            *(dest + this->index_[i]) = *(src + i);
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
    mutable Vector<std::int64_t> index_;
};

}  // namespace merlin

#endif  // MERLIN_PERMUTATION_HPP_
