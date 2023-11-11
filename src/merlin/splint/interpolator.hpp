// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLATOR_HPP_
#define MERLIN_SPLINT_INTERPOLATOR_HPP_

#include <tuple>  // std::tuple

#include "merlin/array/nddata.hpp"        // merlin::array::NdData
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/splint/declaration.hpp"  // merlin::splint::CartesianGrid
#include "merlin/splint/tools.hpp"        // merlin::splint::Method
#include "merlin/synchronizer.hpp"        // merlin::ProcessorType, merlin::Synchronizer
#include "merlin/vector.hpp"              // merlin::Vector, merlin::floatvec

namespace merlin {

namespace splint {

/** @brief Create pointer to copied members for merlin::splint::Interpolator on GPU.*/
void create_intpl_gpuptr(const splint::CartesianGrid & cpu_grid, const Vector<splint::Method> & cpu_methods,
                         splint::CartesianGrid *& gpu_pgrid, Vector<splint::Method> *& gpu_pmethods,
                         std::uintptr_t stream_ptr);

}  // namespace splint

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
     *  @param grid Cartesian grid to interpolate.
     *  @param values C-contiguous array containing the data.
     *  @param method Interpolation method to use for each dimension.
     *  @param n_threads Number of CPU threads to calculate the coefficients.
    */
    MERLIN_EXPORTS Interpolator(const splint::CartesianGrid & grid, const array::Array & values,
                                const Vector<splint::Method> & method, ProcessorType processor = ProcessorType::Cpu);
    /// @}

    /// @name Get elements and attributes
    /// @{
    /** @brief Get reference to the grid.*/
    constexpr splint::CartesianGrid & get_grid(void) noexcept { return *(this->p_grid_); }
    /** @brief Get constant reference to the grid.*/
    constexpr const splint::CartesianGrid & get_grid(void) const noexcept { return *(this->p_grid_); }
    /** @brief Get reference to the coefficient array.*/
    constexpr array::NdData & get_coeff(void) noexcept { return *(this->p_coeff_); }
    /** @brief Get const reference to the coefficient array.*/
    constexpr const array::NdData & get_coeff(void) const noexcept { return *(this->p_coeff_); }
    /** @brief Get reference to the interpolation methods applied to each dimension.*/
    constexpr Vector<splint::Method> & get_method(void) noexcept { return *(this->p_method_); }
    /** @brief Get constant reference to the interpolation methods applied to each dimension.*/
    constexpr const Vector<splint::Method> & get_method(void) const noexcept { return *(this->p_method_); }
    /** @brief Get GPU ID.*/
    constexpr unsigned int gpu_id(void) const noexcept { return this->gpu_id_; }
    /** @brief Check if the interpolator is executed on GPU.*/
    constexpr bool on_gpu(void) const noexcept { return (this->gpu_id_ != static_cast<unsigned int>(-1)); }
    /// @}

    /// @name Construct coefficients
    /// @{
    /** @brief Calculate interpolation coefficients based on provided method.*/
    void build_coefficients(std::uint64_t n_threads = 1);
    /// @}

#ifdef __comment
    /// @name Interpolate on a set of points
    /// @{
    /** @brief Interpolation by CPU.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points to
     *  interpolate and ``ndim`` is the dimension of the point.
     *  @param n_threads Number of CPU threads to evaluate the interpolation.
     */
    floatvec interpolate(const array::Array & points, std::uint64_t n_threads = 1);
    /** @brief Interpolate by GPU.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points to
     *  interpolate and ``ndim`` is the dimension of the point.
     *  @param stream CUDA Stream on which the calculation is launched.
     *  @param n_threads Number of GPU threads to calculate the coefficients.
     */
    floatvec interpolate(const array::Parcel & points, const cuda::Stream & stream = cuda::Stream(),
                         std::uint64_t n_threads = 32);
    /// @}
#endif
    /// @name Synchronization
    /// @{
    /** @brief Force the current CPU to wait until all asynchronous tasks have finished.*/
    void synchronize(void) {
        this->synchronizer_.synchronize();
    }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Interpolator(void);
    /// @}

  private:
    /** @brief Cartesian grid to interpolate.*/
    splint::CartesianGrid * p_grid_ = nullptr;
    /** @brief Pointer to coefficient array.*/
    array::NdData * p_coeff_ = nullptr;
    /** @brief Interpolation method to applied on each dimension.*/
    Vector<splint::Method> * p_method_ = nullptr;
    /** @brief Number of dimension of the interpolation.*/
    std::uint64_t ndim_;
    /** @brief Size of dynamic shared memory needed to perform the calculations on GPU.*/
    std::uint64_t shared_mem_size_ = 0;
    /** @brief GPU ID on which the memory are allocated.*/
    unsigned int gpu_id_;
    /** @brief Synchronizer.*/
    Synchronizer synchronizer_;
};

}  // namespace merlin

#endif  // MERLIN_SPLINT_INTERPOLATOR_HPP_
