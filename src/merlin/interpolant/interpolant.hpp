// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_INTERPOLANT_HPP_
#define MERLIN_INTERPOLANT_INTERPOLANT_HPP_

#include <cstdint>  // std::uint64_t
#include <utility>  // std::exchange

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::NdData, merlin::array::Parcel
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {
class PolynomialInterpolant;

/** @brief Method for polynomial interpolation.*/
enum class Method {
    /** @brief Lagrange method.*/
    Lagrange,
    /** @brief Newton method.*/
    Newton
};

}  // namespace merlin::interpolant

namespace merlin {

/** @brief Polynomial interpolation.*/
class interpolant::PolynomialInterpolant {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    PolynomialInterpolant(void) = default;
    /** @brief Constructor from a Cartesian grid and an array of values using CPU.*/
    PolynomialInterpolant(const interpolant::CartesianGrid & grid, const array::Array & values,
                          interpolant::Method method = interpolant::Method::Lagrange);
    /** @brief Constructor from a Cartesian grid and an array of values using GPU.*/
    PolynomialInterpolant(const interpolant::CartesianGrid & grid, const array::Parcel & values,
                          interpolant::Method method = interpolant::Method::Lagrange,
                          const cuda::Stream & stream = cuda::Stream(),
                          std::uint64_t n_threads = Environment::default_block_size);
    /** @brief Constructor from Sparse grid and an array of values (non-required values may be empty).*/
    PolynomialInterpolant(const interpolant::SparseGrid & grid, const array::Array & values,
                          interpolant::Method method = interpolant::Method::Lagrange);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    PolynomialInterpolant(const interpolant::PolynomialInterpolant & src);
    /** @brief Copy assignment.*/
    PolynomialInterpolant & operator=(const PolynomialInterpolant & src);
    /** @brief Move constructor.*/
    PolynomialInterpolant(PolynomialInterpolant && src) {
        this->grid_ = std::exchange(src.grid_, nullptr);
        this->coeff_ = std::exchange(src.coeff_, nullptr);
        this->method_ = src.method_;
    }
    /** @brief Move assignment.*/
    PolynomialInterpolant & operator=(PolynomialInterpolant && src) {
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
    constexpr interpolant::Grid & get_grid(void) noexcept {return *(this->grid_);}
    /** @brief Get constant reference to interpolating grid.*/
    constexpr const interpolant::Grid & get_grid(void) const noexcept {return *(this->grid_);}
    /** @brief Get reference to array of coefficients.*/
    constexpr array::NdData & get_coeff(void) noexcept {return *(this->coeff_);}
    /** @brief Get constant reference to array of coefficients.*/
    constexpr const array::NdData & get_coeff(void) const noexcept {return *(this->coeff_);}
    /** @brief Get processor.*/
    bool is_calc_on_cpu(void) const;
    /** @brief Get grid type.*/
    bool is_grid_cartesian(void) const;
    /// @}

    /// @name Evaluate interpolation
    /// @{
    /** @brief Evaluate interpolation at a point.
     *  @note Use for CPU interpolant only.
     */
    double operator()(const Vector<double> & point) const;
    /** @brief Evaluate interpolation at multiple points on GPU.
     *  @note Use for GPU interpolant only.
     */
    Vector<double> operator()(const array::Parcel & points, const cuda::Stream & stream = cuda::Stream(),
                              std::uint64_t n_thread = Environment::default_block_size) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    ~PolynomialInterpolant(void);
    /// @}

  protected:
    /** @brief Pointer to grid.*/
    interpolant::Grid * grid_ = nullptr;
    /** @brief Pointer to array of coefficients.*/
    array::NdData * coeff_ = nullptr;
    /** @brief Interpolation method.*/
    interpolant::Method method_;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_INTERPOLANT_HPP_
