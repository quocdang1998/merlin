// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLATOR_HPP_
#define MERLIN_SPLINT_INTERPOLATOR_HPP_

#include <string>   // std::string
#include <tuple>    // std::tuple
#include <utility>  // std::exchange, std::move, std::swap

#include "merlin/array/nddata.hpp"         // merlin::array::NdData
#include "merlin/exports.hpp"              // MERLIN_EXPORTS
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/splint/declaration.hpp"   // merlin::splint::Method
#include "merlin/synchronizer.hpp"         // merlin::Synchronizer
#include "merlin/vector.hpp"               // merlin::Index, merlin::DoubleVec

namespace merlin {

/** @brief Interpolation on a multi-dimensional data.
 *  @details A multi-dimensional interpolation is a linear combination of basis functions. Each basis function is a
 *  product of mono-variable functions.
 *
 *  Thanks to this property, the linear system of equations to solve for the interpolation coefficients can be
 *  decomposed as Kronecker product of one-dimensional matrices, allowing the reduction per dimension of the original
 *  matrix. Furthermore, the process of solving each component matrix can be parallelized over each linear sub-system
 *  created, thus reducing the calculation time of the coefficients.
 */
class splint::Interpolator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Interpolator(void) = default;
    /** @brief Construct from an array of values.
     *  @warning This function will lock the mutex in GPU configuration.
     *  @param grid Cartesian grid to interpolate.
     *  @param values Multi-dimensional array containing the data.
     *  @param p_method Pointer to interpolation method to use for each dimension.
     *  @param synchronizer Asynchronous stream to calculate the interpolation.
     */
    MERLIN_EXPORTS Interpolator(const grid::CartesianGrid & grid, const array::Array & values,
                                const splint::Method * p_method, Synchronizer & synchronizer);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Interpolator(const splint::Interpolator & src) = delete;
    /** @brief Copy assignment.*/
    splint::Interpolator & operator=(const splint::Interpolator & src) = delete;
    /** @brief Move constructor.*/
    Interpolator(splint::Interpolator && src) : ndim_(src.ndim_), shared_mem_size_(src.shared_mem_size_) {
        this->p_grid_ = std::exchange(src.p_grid_, nullptr);
        this->p_coeff_ = std::exchange(src.p_coeff_, nullptr);
        this->p_method_ = std::exchange(src.p_method_, nullptr);
        this->p_synch_ = std::exchange(src.p_synch_, nullptr);
    }
    /** @brief Move assignment.*/
    splint::Interpolator & operator=(splint::Interpolator && src) {
        std::swap(this->p_grid_, src.p_grid_);
        std::swap(this->p_coeff_, src.p_coeff_);
        std::swap(this->p_method_, src.p_method_);
        std::swap(this->p_synch_, src.p_synch_);
        this->ndim_ = src.ndim_;
        this->shared_mem_size_ = src.shared_mem_size_;
        return *this;
    }
    /// @}

    /// @name Get elements and attributes
    /// @{
    /** @brief Get reference to the coefficient array.*/
    constexpr array::NdData & get_coeff(void) noexcept { return *(this->p_coeff_); }
    /** @brief Get const reference to the coefficient array.*/
    constexpr const array::NdData & get_coeff(void) const noexcept { return *(this->p_coeff_); }
    /** @brief Get GPU ID on which the memory is allocated.*/
    constexpr int gpu_id(void) const noexcept {
        if (const cuda::Stream * stream_ptr = std::get_if<cuda::Stream>(&(this->p_synch_->core))) {
            return stream_ptr->get_gpu().id();
        }
        return -1;
    }
    /** @brief Check if the interpolator is executed on GPU.*/
    constexpr bool on_gpu(void) const noexcept { return (this->p_synch_->core.index() == 1); }
    /// @}

    /// @name Construct coefficients
    /// @{
    /** @brief Calculate interpolation coefficients based on provided method.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS void build_coefficients(std::uint64_t n_threads = 1);
    /// @}

    /// @name Interpolate on a set of points
    /// @{
    /** @brief Evaluate interpolation by CPU.
     *  @throw std::invalid_argument when the input array is not C-contiguous.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points to
     *  interpolate and ``ndim`` is the dimension of the point.
     *  @param result Array storing interpolation result.
     *  @param n_threads Number of CPU threads to evaluate the interpolation.
     */
    MERLIN_EXPORTS void evaluate(const array::Array & points, DoubleVec & result, std::uint64_t n_threads = 1);
    /** @brief Evaluate interpolate by GPU.
     *  @warning This function will lock the mutex.
     *  @throw std::invalid_argument when the input array is not C-contiguous.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points to
     *  interpolate and ``ndim`` is the dimension of the point.
     *  @param result Array storing interpolation result.
     *  @param n_threads Number of GPU threads to calculate the coefficients.
     */
    MERLIN_EXPORTS void evaluate(const array::Parcel & points, DoubleVec & result, std::uint64_t n_threads = 32);
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.
     *  @warning This function will lock the mutex.
     */
    MERLIN_EXPORTS ~Interpolator(void);
    /// @}

  protected:
    /** @brief Cartesian grid to interpolate.*/
    grid::CartesianGrid * p_grid_ = nullptr;
    /** @brief Pointer to coefficient array.*/
    array::NdData * p_coeff_ = nullptr;
    /** @brief Interpolation method to applied on each dimension.*/
    Index * p_method_ = nullptr;
    /** @brief Synchronizer.*/
    Synchronizer * p_synch_ = nullptr;

  private:
    /** @brief Number of dimension of the interpolation.*/
    std::uint64_t ndim_;
    /** @brief Size of dynamic shared memory needed to perform the calculations on GPU.*/
    std::uint64_t shared_mem_size_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_SPLINT_INTERPOLATOR_HPP_
