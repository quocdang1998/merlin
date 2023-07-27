// Copyright 2022 quocdang1998
#ifndef MERLIN_INTPL_INTERPOLANT_HPP_
#define MERLIN_INTPL_INTERPOLANT_HPP_

#include <cstdint>  // std::uint64_t
#include <utility>  // std::exchange

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::NdData, merlin::array::Parcel
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/intpl/grid.hpp"  // merlin::intpl::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::intpl {
class PolynomialInterpolant;

/** @brief Method for polynomial interpolation.*/
enum class Method {
    /** @brief Lagrange method.*/
    Lagrange,
    /** @brief Newton method.*/
    Newton
};

}  // namespace merlin::intpl

namespace merlin {

/** @brief Polynomial interpolation.*/
class intpl::PolynomialInterpolant {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    PolynomialInterpolant(void) = default;
    /** @brief Constructor from a Cartesian grid and an array of values using CPU.*/
    MERLIN_EXPORTS PolynomialInterpolant(const intpl::CartesianGrid & grid, const array::Array & values,
                                         intpl::Method method = intpl::Method::Lagrange);
    /** @brief Constructor from a Cartesian grid and an array of values using GPU.*/
    MERLIN_EXPORTS PolynomialInterpolant(const intpl::CartesianGrid & grid, const array::Parcel & values,
                                         intpl::Method method = intpl::Method::Lagrange,
                                         const cuda::Stream & stream = cuda::Stream(),
                                         std::uint64_t n_threads = Environment::default_block_size);
    /** @brief Constructor from Sparse grid and an array of values (non-required values may be empty).*/
    MERLIN_EXPORTS PolynomialInterpolant(const intpl::SparseGrid & grid, const array::Array & values,
                                         intpl::Method method = intpl::Method::Lagrange);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    MERLIN_EXPORTS PolynomialInterpolant(const intpl::PolynomialInterpolant & src);
    /** @brief Copy assignment.*/
    MERLIN_EXPORTS PolynomialInterpolant & operator=(const PolynomialInterpolant & src);
    /** @brief Move constructor.*/
    MERLIN_EXPORTS PolynomialInterpolant(PolynomialInterpolant && src) {
        this->grid_ = std::exchange(src.grid_, nullptr);
        this->coeff_ = std::exchange(src.coeff_, nullptr);
        this->method_ = src.method_;
    }
    /** @brief Move assignment.*/
    MERLIN_EXPORTS PolynomialInterpolant & operator=(PolynomialInterpolant && src) {
        if (this->grid_ != nullptr) {delete this->grid_;}
        if (this->coeff_ != nullptr) {delete this->coeff_;}
        this->grid_ = std::exchange(src.grid_, nullptr);
        this->coeff_ = std::exchange(src.coeff_, nullptr);
        this->method_ = src.method_;
        return *this;
    }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get reference to interpolating grid.*/
    constexpr intpl::Grid & get_grid(void) noexcept {return *(this->grid_);}
    /** @brief Get constant reference to interpolating grid.*/
    constexpr const intpl::Grid & get_grid(void) const noexcept {return *(this->grid_);}
    /** @brief Get reference to array of coefficients.*/
    constexpr array::NdData & get_coeff(void) noexcept {return *(this->coeff_);}
    /** @brief Get constant reference to array of coefficients.*/
    constexpr const array::NdData & get_coeff(void) const noexcept {return *(this->coeff_);}
    /** @brief Get processor.*/
    MERLIN_EXPORTS bool is_calc_on_cpu(void) const;
    /** @brief Get grid type.*/
    MERLIN_EXPORTS bool is_grid_cartesian(void) const;
    /// @}

    /// @name Evaluate interpolation
    /// @{
    /** @brief Evaluate interpolation at a point.
     *  @note Use for CPU intpl only.
     */
    MERLIN_EXPORTS double operator()(const Vector<double> & point) const;
    /** @brief Evaluate interpolation at multiple points on GPU.
     *  @note Use for GPU intpl only.
     */
    MERLIN_EXPORTS Vector<double> operator()(const array::Parcel & points,
                                             const cuda::Stream & stream = cuda::Stream(),
                                             std::uint64_t n_thread = Environment::default_block_size) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~PolynomialInterpolant(void);
    /// @}

  protected:
    /** @brief Pointer to grid.*/
    intpl::Grid * grid_ = nullptr;
    /** @brief Pointer to array of coefficients.*/
    array::NdData * coeff_ = nullptr;
    /** @brief Interpolation method.*/
    intpl::Method method_;
};

}  // namespace merlin

#endif  // MERLIN_INTPL_INTERPOLANT_HPP_
