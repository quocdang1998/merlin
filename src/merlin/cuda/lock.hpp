// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_LOCK_HPP_
#define MERLIN_CUDA_LOCK_HPP_

#include "merlin/cuda_interface.hpp"  // __cudevice__

namespace merlin {

#ifdef __NVCC__
/** @brief Create a critical section allowing only one thread block executes at a time.*/
class KernelLock {
  public:
    /// @name Constructor
    /// @{
    /** @brief Constructor.
     *  @details Allocate an `int` on GPU to indicate wwhether the block are executed or not.
     */
    KernelLock(void) {
        ::cudaMallocAsync(&this->state_, sizeof(int), nullptr);
        int temp = 0;
        ::cudaMemcpy(this->state_, &temp, sizeof(int), ::cudaMemcpyHostToDevice);
        this->reference_count_ = 1;
    }
    /// @}

    /// @name Copy and Move
    /// @{
    /** @brief Copy constructor
     *  @details No new memory is allocated. The copied object points to the same memory as the source.
     */
    KernelLock(const KernelLock & src) : state_(src.state_) { this->reference_count_ = src.reference_count_ + 1; }
    /** @brief Copy assignment.*/
    KernelLock & operator=(const KernelLock & src) {
        this->state_ = src.state_;
        this->reference_count_ = src.reference_count_ + 1;
        return *this;
    }
    /** @brief Move constructor.*/
    KernelLock(KernelLock && src) = delete;
    /** @brief Move assignment.*/
    KernelLock & operator=(KernelLock && src) = delete;
    /// @}

    /// @name Mutex lock
    /// @{
    /** @brief Lock the region (other thread blocks are blocked).*/
    __cudevice__ void lock(void) {
        if (threadIdx.x == 0) {
            while (atomicCAS(this->state_, 0, 1) != 0) {
            }
        }
    }
    /** @brief Unlock the region (other thread blocks can continue the execution).*/
    __cudevice__ void unlock(void) {
        if (threadIdx.x == 0) {
            atomicExch(this->state_, 0);
        }
    }
    /// @}

    /// @name Destructor
    /// @{
    ~KernelLock(void) {
        if (this->reference_count_ == 1) {
            ::cudaFreeAsync(this->state_, nullptr);
        }
    }
    /// @}

  private:
    /** @brief Pointer to the state of the lock on GPU.*/
    int * state_ = nullptr;
    /** @brief Number of locks utilizing the same pointer to the state.*/
    int reference_count_ = 0;
};
#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_CUDA_LOCK_HPP_
