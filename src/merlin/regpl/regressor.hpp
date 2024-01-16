// Copyright 2024 quocdang1998
#ifndef MERLIN_REGPL_REGRESSOR_HPP_
#define MERLIN_REGPL_REGRESSOR_HPP_

#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Stream
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/regpl/declaration.hpp"  // merlin::regpl::Polynomial, merlin::regpl::Regressor
#include "merlin/synchronizer.hpp"  // merlin::ProcessorType, merlin::Synchronizer

namespace merlin {

// Utility
// -------

namespace regpl {

/** @brief Allocate memory for regressor object on GPU.*/
void allocate_mem_gpu(const regpl::Polynomial & polynom, regpl::Polynomial *& p_poly, double *& matrix_data,
                      std::uintptr_t stream_ptr);

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
};

}  // namespace merlin

#endif  // MERLIN_REGPL_REGRESSOR_HPP_
