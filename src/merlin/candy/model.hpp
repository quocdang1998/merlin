// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_MODEL_HPP_
#define MERLIN_CANDY_MODEL_HPP_

#include <cstdint>  // std::uint64_t
#include <string>  // std::string

#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

/** @brief Canonical decomposition model.*/
class candy::Model {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Model(void);
    /** @brief Constructor from shape and rank.*/
    MERLIN_EXPORTS Model(const intvec & shape, std::uint64_t rank);
    /** @brief Constructor from model values.*/
    MERLIN_EXPORTS Model(const Vector<Vector<double>> & parameter, std::uint64_t rank);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS Model(const candy::Model & src) = default;
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS candy::Model & operator=(const candy::Model & src) = default;
    /** @brief Move constructor.*/
    MERLIN_EXPORTS Model(candy::Model && src) = default;
    /** @brief Move assignment.*/
    MERLIN_EXPORTS candy::Model & operator=(candy::Model && src) = default;
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get constant reference to parameters.*/
    __cuhostdev__ constexpr const Vector<Vector<double>> & parameters(void) const noexcept {return this->parameters_;}
    /** @brief Number of dimension.*/
    __cuhostdev__ constexpr std::uint64_t ndim(void) const noexcept {return this->parameters_.size();}
    /** @brief Get shape.*/
    __cuhostdev__ intvec get_model_shape(std::uint64_t * data_ptr = nullptr) const noexcept;
    /** @brief Get rank.*/
    __cuhostdev__ constexpr std::uint64_t rank(void) const noexcept {return this->rank_;}
    /** @brief Size (number of elements).*/
    __cuhostdev__ std::uint64_t size(void) const noexcept;
    /// @}

    /// @name Get and set parameters
    /// @{
    /** @brief Get value of element at a given index.*/
    __cuhostdev__ constexpr const double & get(std::uint64_t i_dim, std::uint64_t index,
                                               std::uint64_t rank) const noexcept {
        return this->parameters_[i_dim][index*this->rank_ + rank];
    }
    /** @brief Set value of element at a given index.*/
    __cuhostdev__ constexpr void set(std::uint64_t i_dim, std::uint64_t index, std::uint64_t rank,
                                     double && value) noexcept {
        this->parameters_[i_dim][index*this->rank_ + rank] = value;
    }
    /** @brief Get reference to element from flattened index.*/
    __cuhostdev__ const double & get(std::uint64_t index) const noexcept;
    /** @brief Set value of element from flattened index.*/
    __cuhostdev__ void set(std::uint64_t index, double && value) noexcept;
    /// @}

    /// @name Evaluation of the model
    /// @{
    /** @brief Evaluate result of the model at a given index.*/
    __cuhostdev__ double eval(const intvec & index) const noexcept;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Model(void);
    /// @}

  protected:
    /** @brief Pointer to values of parameters.*/
    Vector<Vector<double>> parameters_;
    /** @brief Rank of the decomposition.*/
    std::uint64_t rank_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_CANDY_MODEL_HPP_
