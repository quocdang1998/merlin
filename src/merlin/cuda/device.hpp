// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_DEVICE_HPP_
#define MERLIN_CUDA_DEVICE_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t
#include <limits>   // std::numeric_limits
#include <string>   // std::string

#include "merlin/config.hpp"             // __cuhostdev__
#include "merlin/cuda/declaration.hpp"   // merlin::cuda::Device
#include "merlin/cuda/enum_wrapper.hpp"  // merlin::cuda::DeviceLimit
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
    /** @brief Push the primary context associated to the GPU to the context stack.
     *  @warning This function will also lock the Environment::mutex without releasing it. User is reponsible for
     *  ensuring that the mutex is not locked by the current thread before calling this function (otherwise the
     *  situation will result in an infinite loop). Releasing the mutex after usage is automatically called inside the
     *  method cuda::Device::pop_context.
     *  @return Pointer to old context.
     */
    MERLIN_EXPORTS std::uintptr_t push_context(void) const;
    /** @brief Pop the current context out of the context stack.
     *  @warning This function will also unlock the mutex.
     */
    MERLIN_EXPORTS static void pop_context(std::uintptr_t previous_context);
    /** @brief Get and set setting limits of the current GPU.
     *  @return Value of the limit of the current GPU if argument ``size`` is not given, and the value of size
     *  otherwise.
     */
    MERLIN_EXPORTS static std::uint64_t limit(cuda::DeviceLimit limit,
                                              std::uint64_t size = std::numeric_limits<std::uint64_t>::max());
    /** @brief Reset GPU.
     *  @details Destroy all allocations and reset the state of the current GPU.
     */
    MERLIN_EXPORTS static void reset_all(void);
    /** @brief Synchronize the current GPU.*/
    MERLIN_EXPORTS static void synchronize(void);
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

class cuda::CtxGuard {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    CtxGuard(void) = default;
    /** @brief Contructor from cuda::Device.*/
    CtxGuard(const cuda::Device & gpu) { this->context_ptr_ = gpu.push_context(); }
    /// @}

    /// @name Destructor
    /** @brief Default destructor.*/
    inline ~CtxGuard(void) {
        if (this->context_ptr_ != 0) {
            cuda::Device::pop_context(this->context_ptr_);
        }
    }

  protected:
    /** @brief Pointer to the context guarded.*/
    std::uintptr_t context_ptr_ = 0;
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
