// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_REGRESSOR_HPP_
#define MERLIN_REGPL_REGRESSOR_HPP_

#include <utility>  // std::exchange

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Stream
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid, merlin::grid::RegularGrid
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Polynomial, merlin::regpl::Regressor
#include "merlin/synchronizer.hpp"  // merlin::ProcessorType, merlin::Synchronizer
#include "merlin/vector.hpp"  // merlin::floatvec

namespace merlin {

// Utility
// -------

namespace regpl {

/** @brief Allocate memory for regressor object on GPU.*/
void allocate_mem_gpu(const regpl::Polynomial & polynom, regpl::Polynomial *& p_poly, double *& matrix_data,
                      std::uintptr_t stream_ptr);

/** @brief Calculate coefficient for regression on Cartesian grid.*/
void fit_cartgrid_by_cpu(std::shared_future<void> synch, regpl::Polynomial * p_poly, double * matrix_data,
                         const grid::CartesianGrid * p_grid, const array::Array * p_data, std::uint64_t n_threads,
                         char * cpu_buffer) noexcept;

/** @brief Calculate coefficient for regression on regular grid.*/
void fit_reggrid_by_cpu(std::shared_future<void> synch, regpl::Polynomial * p_poly, double * matrix_data,
                        const grid::RegularGrid * p_grid, const floatvec * p_data, std::uint64_t n_threads,
                        char * cpu_buffer) noexcept;

/** @brief Evaluate regression by CPU.*/
void eval_by_cpu(std::shared_future<void> synch, const regpl::Polynomial * p_poly, const array::Array * p_data,
                 double * p_result, std::uint64_t n_threads, char * cpu_buffer) noexcept;

/** @brief Evaluate regression by GPU.*/
void eval_by_gpu(const regpl::Polynomial * p_poly, const double * p_data, double * p_result, std::uint64_t n_points,
                 std::uint64_t ndim, std::uint64_t shared_mem_size, std::uint64_t n_threads,
                 const cuda::Stream & stream) noexcept;

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
    /** @brief Constructor from polynomial object.*/
    MERLIN_EXPORTS Regressor(const regpl::Polynomial & polynom, ProcessorType proc_type = ProcessorType::Cpu);
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor.*/
    Regressor(const Regressor & src) = delete;
    /** @brief Copy assignment.*/
    regpl::Regressor & operator=(const Regressor & src) = delete;
    /** @brief Move constructor.*/
    Regressor(Regressor && src) {
        this->p_poly_ = std::exchange(src.p_poly_, nullptr);
        this->matrix_data_ = std::exchange(src.matrix_data_, nullptr);
        this->cpu_buffer_ = std::exchange(src.cpu_buffer_, nullptr);
        this->synch_ = std::move(src.synch_);
        this->num_coeff_ = src.num_coeff_;
        this->ndim_ = src.ndim_;
        this->cpu_buffer_size_ = src.cpu_buffer_size_;
        this->shared_mem_size_ = src.shared_mem_size_;
    }
    /** @brief Move assignment.*/
    regpl::Regressor & operator=(Regressor && src) = delete;
    /// @}

    /// @name Get elements and attributes
    /// @{
    /** @brief Get GPU ID on which the memory is allocated.*/
    constexpr unsigned int gpu_id(void) const noexcept {
        if (const cuda::Stream * stream_ptr = std::get_if<cuda::Stream>(&(this->synch_.synchronizer))) {
            return stream_ptr->get_gpu().id();
        }
        return static_cast<unsigned int>(-1);
    }
    /** @brief Check if the interpolator is executed on GPU.*/
    constexpr bool on_gpu(void) const noexcept { return (this->synch_.synchronizer.index() == 1); }
    /** @brief Get a copy of the polynomial.*/
    MERLIN_EXPORTS regpl::Polynomial get_polynom(void) const;
    /// @}

    /// @name Fit data
    /// @{
    /** @brief Regression on a cartesian dataset using CPU parallelism.
     *  @param grid Cartesian grid of points.
     *  @param data Data to fit.
     *  @param n_threads Number of threads for calculation.
    */
    MERLIN_EXPORTS void fit_cpu(const grid::CartesianGrid & grid, const array::Array & data,
                                std::uint64_t n_threads = 1);
    /** @brief Regression on a random dataset using CPU parallelism.
     *  @param grid Grid of points.
     *  @param data Data to fit.
     *  @param n_threads Number of threads for calculation.
    */
    MERLIN_EXPORTS void fit_cpu(const grid::RegularGrid & grid, const floatvec & data, std::uint64_t n_threads = 1);
    /// @}

    /// @name Evaluate
    /// @{
    /** @brief Evaluate regression by CPU parallelism.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points
     *  and ``ndim`` is the dimension of each point.
     *  @param p_result Pointer to array storing result, size of at least ``double[npoint]``.
     *  @param n_threads Number of threads for calculation.
     */
    void evaluate(const array::Array & points, double * p_result, std::uint64_t n_threads = 1);
    /** @brief Evaluate regression by GPU parallelism.
     *  @param points 2D C-contiguous array of shape ``[npoint, ndim]``, in which ``npoint`` is the number of points
     *  and ``ndim`` is the dimension of each point.
     *  @param p_result Pointer to array storing result, size of at least ``double[npoint]``.
     *  @param n_threads Number of threads for calculation.
     */
    void evaluate(const array::Parcel & points, double * p_result, std::uint64_t n_threads = 32);
    /// @}

    /// @name Synchronization
    /// @{
    /** @brief Force the current CPU to wait until all asynchronous tasks have finished.*/
    void synchronize(void) { this->synch_.synchronize(); }
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    MERLIN_EXPORTS ~Regressor(void);
    /// @}

  protected:
    /** @brief Polynomial object.*/
    regpl::Polynomial * p_poly_ = nullptr;
    /** @brief Memory for the regression matrix.*/
    double * matrix_data_ = nullptr;

    /** @brief Number of coefficients of the polynomial.*/
    std::uint64_t num_coeff_;
    /** @brief Number of dimension polynomial.*/
    std::uint64_t ndim_;
    /** @brief Synchronizer.*/
    Synchronizer synch_;

    /** @brief Buffer memory for CPU calculation.*/
    char * cpu_buffer_ = nullptr;
    /** @brief Size of buffer memory for CPU calculation.*/
    std::uint64_t cpu_buffer_size_;

    /** @brief Shared memory for calculation on GPU.*/
    std::uint64_t shared_mem_size_;

  private:
    /** @brief Resize CPU buffer.*/
    void resize_cpu_buffer(std::uint64_t new_size);
};

}  // namespace merlin

#endif  // MERLIN_REGPL_REGRESSOR_HPP_
