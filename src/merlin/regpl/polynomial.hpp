// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_POLYNOMIAL_HPP_
#define MERLIN_REGPL_POLYNOMIAL_HPP_

#include <string>  // std::string

#include "merlin/array/declaration.hpp"  // merlin::array::Array
#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Polynomial
#include "merlin/vector.hpp"  // merlin::floatvec, merlin::intvec

namespace merlin {

/** @brief Multi-variate polynomial.*/
class regpl::Polynomial {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Polynomial(void) = default;
    /** @brief Constructor of an empty polynomial from order per dimension.*/
    MERLIN_EXPORTS Polynomial(const intvec & order_per_dim);
    /** @brief Constructor of a pre-allocated array of coefficients and order per dimension.*/
    MERLIN_EXPORTS Polynomial(double * coeff_data, const intvec & order_per_dim);
    /// @}

    /// @name Copy and move
    /// @{
    /** @brief Copy constructor.*/
    Polynomial(const regpl::Polynomial & src) = default;
    /** @brief Copy assignment.*/
    regpl::Polynomial & operator=(const regpl::Polynomial & src) = default;
    /** @brief Move constructor.*/
    Polynomial(regpl::Polynomial && src) = default;
    /** @brief Move assignment.*/
    regpl::Polynomial & operator=(regpl::Polynomial && src) = default;
    /// @}

    /// @name Get attributes
    /// @{
    /** @brief Get number of terms in the polynomial.*/
    __cuhostdev__ constexpr const std::uint64_t & size(void) const noexcept { return this->coeff_.size(); }
    /** @brief Get number of dimension.*/
    __cuhostdev__ constexpr const std::uint64_t & ndim(void) const noexcept { return this->order_.size(); }
    /** @brief Get reference to coefficient array of the polynomial.*/
    __cuhostdev__ constexpr floatvec & coeff(void) noexcept { return this->coeff_; }
    /** @brief Get constant reference to coefficient array of the polynomial.*/
    __cuhostdev__ constexpr const floatvec & coeff(void) const noexcept { return this->coeff_; }
    /** @brief Get order per dimension of the polynomial.*/
    __cuhostdev__ constexpr const intvec & order(void) const noexcept { return this->order_; }
    /// @}

    /// @name Vandermonde
    /// @{
    /** @brief Generate Vandermonde matrix.
     *  @param grid_points Points of a grid.
     *  @param n_threads Number of threads for calculating the Vandermonde matrix.
    */
    MERLIN_EXPORTS array::Array calc_vandermonde(const array::Array & grid_points, std::uint64_t n_threads = 1) const;
    /// @}

    /// @name Evaluation
    /// @{
    /** @brief Evaluate polynomial value at a given point.
     *  @param point Pointer to coordinates of the point.
     *  @param buffer Buffer memory for calculation, size of at least ``double[ndim]``,
    */
    __cuhostdev__ double eval(const double * point, double * buffer) const noexcept;
    /// @}

    /// @name GPU related features
    /// @{
    /** @brief Calculate the minimum number of bytes to allocate in the memory to store the polynomial and its data.*/
    std::uint64_t cumalloc_size(void) const noexcept {
        std::uint64_t size = sizeof(regpl::Polynomial);
        size += this->size() * sizeof(double) + this->ndim() * sizeof(std::uint64_t);
        return size;
    }
    /** @brief Copy the polynomial from CPU to a pre-allocated memory on GPU.
     *  @details Values of vectors should be copied to the memory region that comes right after the copied object.
     *  @param gpu_ptr Pointer to a pre-allocated GPU memory holding an instance.
     *  @param coeff_data_ptr Pointer to a pre-allocated GPU memory storing data of coefficients.
     *  @param stream_ptr Pointer to CUDA stream for asynchronious copy.
     */
    MERLIN_EXPORTS void * copy_to_gpu(regpl::Polynomial * gpu_ptr, void * coeff_data_ptr,
                                      std::uintptr_t stream_ptr = 0) const;
    /** @brief Calculate the minimum number of bytes to allocate in CUDA shared memory to store the polynomial.*/
    std::uint64_t sharedmem_size(void) const noexcept { return this->cumalloc_size(); }
#ifdef __NVCC__
    /** @brief Copy polynomial to pre-allocated memory region by current CUDA block of threads.
     *  @details The copy action is performed by the whole CUDA thread block.
     *  @param dest_ptr Memory region where the polynomial is copied to.
     *  @param coeff_data_ptr Pointer to a pre-allocated GPU memory storing data of coefficients, size of
     *  ``std::uint64_t[this->ndim()] + double[this->size()]``.
     *  @param thread_idx Flatten ID of the current CUDA thread in the block.
     *  @param block_size Number of threads in the current CUDA block.
     */
    __cudevice__ void * copy_by_block(regpl::Polynomial * dest_ptr, void * coeff_data_ptr, std::uint64_t thread_idx,
                                      std::uint64_t block_size) const;
    /** @brief Copy polynomial to a pre-allocated memory region by a single GPU threads.
     *  @param dest_ptr Memory region where the polynomial is copied to.
     *  @param coeff_data_ptr Pointer to a pre-allocated GPU memory storing data of coefficients, size of
     *  ``std::uint64_t[this->ndim()] + double[this->size()]``.
     */
    __cudevice__ void * copy_by_thread(regpl::Polynomial * dest_ptr, void * coeff_data_ptr) const;
#endif  // __NVCC__
    /** @brief Copy data from GPU back to CPU.*/
    MERLIN_EXPORTS void * copy_from_gpu(double * data_from_gpu, std::uintptr_t stream_ptr = 0) noexcept;
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    ~Polynomial(void) = default;
    /// @}

  protected:
    /** @brief Coefficient data.*/
    floatvec coeff_;
    /** @brief Order per dimension.*/
    intvec order_;
};

}  // namespace merlin

#endif  // MERLIN_REGPL_POLYNOMIAL_HPP_
