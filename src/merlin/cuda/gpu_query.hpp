// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_GPU_QUERY_HPP_
#define MERLIN_CUDA_GPU_QUERY_HPP_

#include <cstdint>  // std::uint64_t, UINT64_MAX
#include <map>  // std::map
#include <string>  // std::string

#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::cuda {

/** @brief Class representing CPU device.*/
class MERLIN_EXPORTS Device {
  public:
    /** @brief Limit to get.*/
    enum class Limit {
        /** @brief Size of the stack of each CUDA thread.*/
        StackSize = 0x00,
        /** @brief Size of the ``std::printf`` function buffer.*/
        PrintfSize = 0x01,
        /** @brief Size of the heap of each CUDA thread.*/
        HeapSize = 0x02,
        /** @brief Maximum nesting depth of a grid at which a thread can safely call ``cudaDeviceSynchronize``.*/
        SyncDepth = 0x03,
        /** @brief Maximum number of outstanding device runtime launches.*/
        LaunchPendingCount = 0x04
    };

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
    __cuhostdev__ Device(const Device & src) {this->id_ = src.id_;}
    /** @brief Copy assignment.*/
    __cuhostdev__ Device & operator=(const Device & src) {
        this->id_ = src.id_;
        return *this;
    }
    /// @}

    /// @name GPU query
    /// @{
    /** @brief Get current GPU.*/
    __cuhostdev__ static Device get_current_gpu(void);
    /** @brief Get total number of CUDA capable GPU.*/
    __cuhostdev__ static int get_num_gpu(void);
    /** @brief Print GPU specifications.*/
    void print_specification(void);
    /** @brief Test functionality of GPU.
     *  @details This function tests if the installed CUDA is compatible with the GPU driver by perform an addition of
     *  two integers on GPU.
     */
    bool test_gpu(void);
    /// @}

    /// @name GPU action
    /// @{
    /** @brief Set device ass current device.*/
    void set_as_current(void) const;
    /** @brief Get and set limit.
     *  @return Value of current limit if argument ``size`` is not given, and the value of size otherwise.
     */
    static std::uint64_t limit(Limit limit, std::uint64_t size = UINT64_MAX);
    /** @brief Reset GPU.
     *  @details Destroy all allocations and reset all state on the current device in the current process.
     */
    static void reset_all(void);
    /** @brief Compare 2 GPU.*/
    friend bool operator==(const Device & left, const Device & right) {return left.id_ == right.id_;}
    /** @brief Compare 2 GPU.*/
    friend bool operator!=(const Device & left, const Device & right) {return left.id_ != right.id_;}
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to GPU ID.*/
    __cuhostdev__ int & id(void) {return this->id_;}
    /** @brief Get constant reference to GPU ID.*/
    __cuhostdev__ const int & id(void) const {return this->id_;}
    /// @}

    /// @name Representation
    /// @{
    /** @brief String representation.*/
    std::string repr(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Default destructor.*/
    __cuhostdev__ ~Device(void);
    /// @}

  private:
    /** @brief ID of device.*/
    int id_ = -1;
};

/** @brief Print GPU specifications.
 *  @details Print GPU specifications (number of threads, total global memory, max shared memory) and API limitation
 *  (max thread per block, max block per grid).
 */
MERLIN_EXPORTS void print_all_gpu_specification(void);

/** @brief Perform an addition of two integers on GPU.
 *  @details This function tests if the installed CUDA is compatible with the GPU.
 *  @return ``true`` if all tests on all GPU pass.
 */
MERLIN_EXPORTS bool test_all_gpu(void);

}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_GPU_QUERY_HPP_
