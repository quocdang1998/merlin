// Copyright 2023 quocdang1998
#ifndef MERLIN_SHUFFLE_HPP_
#define MERLIN_SHUFFLE_HPP_

#include <random>  // std::mt19937_64
#include <string>  // std::string

#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec

namespace merlin {

/** @brief %Shuffle operation on multi-dimensional object.*/
class Shuffle {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    Shuffle(void) = default;
    /** @brief Constructor from shape vector.*/
    Shuffle(const intvec & shape);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Shuffle(const Shuffle & src) = default;
    /** @brief Copy assignment.*/
    Shuffle & operator=(const Shuffle & src) = default;
    /** @brief Move constructor.*/
    Shuffle(Shuffle && src) = default;
    /** @brief Move assignment.*/
    Shuffle & operator=(Shuffle && src) = default;
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get number of dimension.*/
    constexpr std::uint64_t ndim(void) const noexcept { return this->shuffled_index_.size(); }
    /** @brief Get constant reference to vector of shuffled indexes.*/
    constexpr const Vector<intvec> & shuffled_index(void) const noexcept { return this->shuffled_index_; }
    /// @}

    /// @name Operations
    /// @{
    /** @brief Set random seed*/
    static void set_random_seed(std::uint64_t seed);
    /** @brief Get shuffled index.*/
    intvec operator[](const intvec original_index) const noexcept;
    /** @brief Get inverse permutation.*/
    Shuffle inverse(void) const;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    ~Shuffle(void) = default;
    /// @}

  protected:
    /** @brief Shuffled indexes of each dimension.
     *  @details Each element indicates where the element will be in the resulted element. For example, the j-th
     *  element of i-th dimension will be move to position ``shuffled_index_[i][j]``.
     */
    Vector<intvec> shuffled_index_;
    /** @brief Random number generator engine.
     *  @sa C++ implementation of <a href="https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine">
     *  Mersenne twister engine</a>.
     */
    static std::mt19937_64 & random_generator_;
};

}  // namespace merlin

#endif  // MERLIN_SHUFFLE_HPP_
