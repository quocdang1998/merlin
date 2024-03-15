// Copyright 2023 quocdang1998
#ifndef MERLIN_PERMUTATION_HPP_
#define MERLIN_PERMUTATION_HPP_

#include <string>  // std::string

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::intvec

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
    MERLIN_EXPORTS Permutation(const intvec & index);
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
    /** @brief Reference to index vector.*/
    constexpr Vector<std::int64_t> & index(void) noexcept { return this->index_; }
    /** @brief Constant reference to index vector.*/
    constexpr const Vector<std::int64_t> & index(void) const noexcept { return this->index_; }
    /** @brief Get number of elements permuted.*/
    constexpr std::uint64_t size(void) const noexcept { return this->index_.size(); }
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
