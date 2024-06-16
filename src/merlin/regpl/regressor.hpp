// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_REGRESSOR_HPP_
#define MERLIN_REGPL_REGRESSOR_HPP_

#include <utility>  // std::forward

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Regressor
#include "merlin/regpl/polynomial.hpp"   // merlin::regpl::Polynomial
#include "merlin/synchronizer.hpp"       // merlin::Synchronizer

namespace merlin {

// Utility
// -------

namespace regpl {

/** @brief Evaluate regression by CPU.*/
void eval_by_cpu(std::future<void> && synch, const regpl::Polynomial * p_poly, const double * point_data,
                 double * p_result, std::uint64_t n_points, std::uint64_t n_threads) noexcept;

/** @brief Evaluate regression by GPU.*/
void eval_by_gpu(const regpl::Polynomial * p_poly, const double * point_data, double * p_result, std::uint64_t n_points,
                 std::uint64_t n_threads, std::uint64_t shared_mem_size, const cuda::Stream & stream) noexcept;

}  // namespace regpl

// Regressor
// ---------

/** @brief Launch polynomial regression.*/
class regpl::Regressor {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Regressor(void) = default;
    /** @brief Constructor from polynomial object.
     *  @param polynom Target polynomial to evaluate.
     *  @param synchronizer Reference to a synchronizer.
     */
    Regressor(regpl::Polynomial && polynom, Synchronizer & synchronizer) :
    poly_(std::forward<regpl::Polynomial>(polynom)), p_synch_(&synchronizer) {}
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Regressor(const Regressor & src) = delete;
    /** @brief Copy assignment.*/
    regpl::Regressor & operator=(const Regressor & src) = delete;
    /** @brief Move constructor.*/
    Regressor(Regressor && src) = default;
    /** @brief Move assignment.*/
    regpl::Regressor & operator=(Regressor && src) = default;
    /// @}

    /// @name Get elements and attributes
    /// @{
    /** @brief Get GPU ID on which the memory is allocated.*/
    constexpr unsigned int gpu_id(void) const noexcept {
        if (const cuda::Stream * stream_ptr = std::get_if<cuda::Stream>(&(this->p_synch_->core))) {
            return stream_ptr->get_gpu().id();
        }
        return static_cast<unsigned int>(-1);
    }
    /** @brief Check if the interpolator is executed on GPU.*/
    constexpr bool on_gpu(void) const noexcept { return (this->p_synch_->core.index() == 1); }
    /** @brief Get reference to the polynomial.*/
    regpl::Polynomial & polynom(void) { return this->poly_; }
    /** @brief Get constant reference to the polynomial.*/
    const regpl::Polynomial & polynom(void) const { return this->poly_; }
    /// @}

    /// @name Evaluate
    /// @{
    /** @brief Evaluate regression by CPU parallelism.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points
     *  and ``ndim`` is the dimension of each point.
     *  @param result Array storing result, size of at least ``double[npoint]``.
     *  @param n_threads Number of threads for calculation.
     */
    MERLIN_EXPORTS void evaluate(const array::Array & points, DoubleVec & result, std::uint64_t n_threads = 1);
    /** @brief Evaluate regression by GPU parallelism.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points
     *  and ``ndim`` is the dimension of each point.
     *  @param result Array storing result, size of at least ``double[npoint]``.
     *  @param n_threads Number of threads for calculation.
     */
    MERLIN_EXPORTS void evaluate(const array::Parcel & points, DoubleVec & result, std::uint64_t n_threads = 32);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~Regressor(void) = default;
    /// @}

  protected:
    /** @brief Polynomial object.*/
    regpl::Polynomial poly_;
    /** @brief Synchronizer.*/
    Synchronizer * p_synch_;
};

}  // namespace merlin

#endif  // MERLIN_REGPL_REGRESSOR_HPP_
