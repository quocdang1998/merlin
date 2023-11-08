// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_DEVICE_HPP_
#define MERLIN_CUDA_DEVICE_HPP_

#include <cstdint>  // std::uint64_t, UINT64_MAX
#include <string>   // std::string

#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Device
#include "merlin/cuda/enum_wrapper.hpp"  // merlin::cuda::DeviceLimit
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS

namespace merlin {

/** @brief GPU device.
 *  @details Each GPU attached to the system is identified by an integer ranging from 0 (default GPU) to the number of
 *  GPUs. This class is a thin C++ wrapper around CUDA operations on GPU devices.
 */
class cuda::Device {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ Device(void) {}
    /** @brief Constructor from GPU ID.*/
    __cuhostdev__ Device(int id);
    /// @}

    /// @name Copy and Move
    /// @details Move constructor and Move assignment are deleted because they are not necessary.
    /// @{
    /** @brief Copy constructor.*/
    __cuhostdev__ Device(const cuda::Device & src) { this->id_ = src.id_; }
    /** @brief Copy assignment.*/
    __cuhostdev__ cuda::Device & operator=(const cuda::Device & src) {
        this->id_ = src.id_;
        return *this;
    }
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to GPU ID.*/
    __cuhostdev__ constexpr int & id(void) noexcept { return this->id_; }
    /** @brief Get constant reference to GPU ID.*/
    __cuhostdev__ constexpr const int & id(void) const noexcept { return this->id_; }
    /// @}

    /// @name Query
    /// @{
    /** @brief Get current GPU.*/
    __cuhostdev__ static cuda::Device get_current_gpu(void);
    /** @brief Get total number of CUDA capable GPU.*/
    __cuhostdev__ static std::uint64_t get_num_gpu(void);
    /** @brief Print GPU specifications.*/
    MERLIN_EXPORTS void print_specification(void) const;
    /** @brief Test functionality of GPU.
     *  @details This function tests if the installed CUDA is compatible with the GPU driver by perform an addition of
     *  two integers on the specified GPU.
     */
    MERLIN_EXPORTS bool test_gpu(void) const;
    /// @}

    /// @name Action
    /// @{
    /** @brief Set the GPU as current device.
     *  @details Replace the current CUDA context by the primary context associated the GPU.
     */
    MERLIN_EXPORTS void set_as_current(void) const;
    /** @brief Get and set setting limits of the current GPU.
     *  @return Value of the limit of the current GPU if argument ``size`` is not given, and the value of size
     *  otherwise.
     */
    MERLIN_EXPORTS static std::uint64_t limit(cuda::DeviceLimit limit, std::uint64_t size = UINT64_MAX);
    /** @brief Reset GPU.
     *  @details Destroy all allocations and reset the state of the current GPU.
     */
    MERLIN_EXPORTS static void reset_all(void);
    /// @}

    /// @name Comparison
    /// @{
    /** @brief Identical comparison operator.*/
    friend bool constexpr operator==(const cuda::Device & left, const cuda::Device & right) noexcept {
        return left.id_ == right.id_;
    }
    /** @brief Different comparison operator.*/
    friend bool constexpr operator!=(const cuda::Device & left, const cuda::Device & right) noexcept {
        return left.id_ != right.id_;
    }
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    MERLIN_EXPORTS std::string str(void) const;
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    __cuhostdev__ ~Device(void);
    /// @}

  protected:
    /** @brief ID of device.*/
    int id_ = -1;
};

namespace cuda {

/** @brief Perform addition of 2 integers on GPU.
 *  @param p_a Pre-allocated pointer on GPU to the first summand.
 *  @param p_b Pre-allocated pointer on GPU to the second summand.
 *  @param p_result Pre-allocated pointer on GPU to the result;.
 */
void add_integers_on_gpu(int * p_a, int * p_b, int * p_result);

/** @brief Print GPU specifications.
 *  @details Print GPU specifications (number of threads, total global memory, max shared memory) and API limitation
 *  (max thread per block, max block per grid) of all CUDA capable GPUs.
 */
MERLIN_EXPORTS void print_gpus_spec(void);

/** @brief Test if the compiled library is compatible with all CUDA capable GPUs.
 *  @details Perform an addition of two integers on each CUDA capable GPU.
 *  @return ``true`` if all tests on all GPU pass.
 */
MERLIN_EXPORTS bool test_all_gpu(void);

}  // namespace cuda

}  // namespace merlin

#endif  // MERLIN_CUDA_DEVICE_HPP_
